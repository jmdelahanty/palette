"""Shared ROI crop reader for materialized and geometry-only crop runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.composite_crop import (
    COMPOSITE_CROP_STORAGE_MODE,
    CompositeCropArray,
)
from fisheye.shared.crop_roi_layout import (
    DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
    DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
    SCRATCH_ROI_CACHE_LAYOUT_PROFILE,
    build_crop_roi_create_kwargs,
    build_scratch_roi_cache_layout,
    crop_roi_layout_attrs,
)
from fisheye.shared.flat_roi_cache import (
    crop_run_name_from_manifest,
    open_flat_roi_cache,
)
from fisheye.shared.grayscale import rgb_to_gray_bt601_cv2_uint8
from fisheye.shared.roi_pixel_contract import (
    ROI_IMAGE_REPRESENTATION,
    SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
    SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME,
    SOURCE_PIXELS_RAW_CAMERA_VIDEO,
    crop_image_source_live_pixel_contract,
    crop_run_pixel_contract,
    normalize_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.source_video_metadata import (
    SourceVideoMetadataMissingError,
    resolve_source_video,
)
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr.crop_consumer import (
    authoritative_crop_roi_pixel_contract,
    build_crop_run_reference,
    strict_crop_fixed_roi_shape,
    strict_crop_source_frame_shape,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_ATTRIBUTE,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
)

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")

try:  # pragma: no cover - optional runtime dependency
    import decord  # type: ignore
    from decord import VideoReader, cpu  # type: ignore

    _DECORD_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    decord = None  # type: ignore
    VideoReader = None  # type: ignore
    cpu = None  # type: ignore
    _DECORD_IMPORT_ERROR = exc

try:  # pragma: no cover - optional runtime dependency
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - import itself is environment dependent
    _CV2_IMPORT_ERROR = None

try:  # pragma: no cover - optional runtime dependency
    from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader  # type: ignore

    _PYNVVC_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    PynvvcLumaRgbReader = None  # type: ignore
    _PYNVVC_IMPORT_ERROR = exc

_ROI_CACHE_POLICIES = {"never", "auto", "always"}
_ROI_CACHE_AUTO_MIN_SOURCE_PIXELS = 2048 * 2048
_ROI_CACHE_BUILD_BATCH = 64
_ROI_LIVE_ACCELERATION_CHOICES = {"auto", "cpu", "gpu"}
_ROI_LIVE_GPU_CHUNK_FRAMES_DEFAULT = 32


def _gpu_decode_unavailable(reason: str) -> RuntimeError:
    return RuntimeError(
        "GPU decode unavailable; refusing CPU fallback - pixels would differ from the "
        f"production path ({reason})"
    )


def _normalize_run_name(value: object) -> str | None:
    normalized = normalize_attr(value)
    if normalized is None:
        return None
    return str(normalized) or None


def _available_group_names(parent: zarr.Group) -> list[str]:
    if hasattr(parent, "group_keys"):
        names = parent.group_keys()
    else:  # pragma: no cover - defensive fallback for fake groups
        names = parent.keys()
    return sorted(str(name) for name in names)


def _open_crop_parent(root: zarr.Group, zarr_path: Path | None = None) -> zarr.Group:
    if zarr_path is not None:
        crop_parent_path = zarr_path / "crop_runs"
        if crop_parent_path.exists():
            try:
                return zarr.open_group(str(crop_parent_path), mode="r")
            except Exception:
                pass
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise ValueError("Zarr archive is missing crop_runs; run cropping first")
    return crop_parent


def resolve_crop_run(
    root: zarr.Group,
    *,
    crop_run: str | None = None,
    zarr_path: str | Path | None = None,
) -> tuple[zarr.Group, zarr.Group, str]:
    """Resolve a crop run for mixed-mode readers.

    Resolution order is:
    1. explicit ``crop_run``
    2. ``latest_any``
    3. legacy/backward-compatible ``latest``
    4. ``latest_materialized``
    """

    resolved_zarr_path = Path(zarr_path) if zarr_path is not None else None
    crop_parent = _open_crop_parent(root, resolved_zarr_path)
    requested = _normalize_run_name(crop_run)
    available = _available_group_names(crop_parent)

    if requested is not None:
        if requested not in crop_parent:
            available_text = ", ".join(available) or "(none)"
            raise ValueError(
                f"Crop run '{requested}' not found under crop_runs. Available: {available_text}"
            )
        return crop_parent, crop_parent[requested], requested

    for attr_name in ("latest_any", "latest", "latest_materialized"):
        candidate = _normalize_run_name(crop_parent.attrs.get(attr_name))
        if candidate and candidate in crop_parent:
            run_group = crop_parent[candidate]
            if is_run_selector_eligible(run_group) and is_run_complete_in_parent(
                crop_parent,
                run_group,
            ):
                return crop_parent, run_group, candidate

    raise ValueError(
        "No crop run found; cannot resolve latest_any/latest/latest_materialized"
    )


def resolve_materialized_crop_run(
    root: zarr.Group,
    *,
    crop_run: str | None = None,
    zarr_path: str | Path | None = None,
) -> tuple[zarr.Group, zarr.Group, str]:
    """Resolve a crop run that is guaranteed to provide materialized ROI images.

    Traditional pipelines intentionally depend on persisted ROI pixels and
    should not silently fall back to geometry-only runs.
    """

    resolved_zarr_path = Path(zarr_path) if zarr_path is not None else None
    crop_parent = _open_crop_parent(root, resolved_zarr_path)
    available = _available_group_names(crop_parent)
    requested = _normalize_run_name(crop_run)

    def _ensure_materialized(run_name: str) -> tuple[zarr.Group, zarr.Group, str]:
        run_group = crop_parent[run_name]
        storage_mode = _resolve_storage_mode(run_group)
        if storage_mode != "materialized" or "roi_images" not in run_group:
            detail = (
                "is composite and must be resolved through CropImageSource"
                if storage_mode == COMPOSITE_CROP_STORAGE_MODE
                else "is not materialized"
            )
            raise ValueError(
                f"Crop run '{run_name}' {detail}. Traditional pipelines require "
                "crop_runs/<run>/roi_images."
            )
        return crop_parent, run_group, run_name

    if requested is not None:
        if requested not in crop_parent:
            available_text = ", ".join(available) or "(none)"
            raise ValueError(
                f"Crop run '{requested}' not found under crop_runs. Available: {available_text}"
            )
        return _ensure_materialized(requested)

    for attr_name in ("latest_materialized", "latest"):
        candidate = _normalize_run_name(crop_parent.attrs.get(attr_name))
        if candidate and candidate in crop_parent:
            run_group = crop_parent[candidate]
            if not is_run_selector_eligible(run_group) or not is_run_complete_in_parent(
                crop_parent,
                run_group,
            ):
                continue
            if (
                _resolve_storage_mode(run_group) == "materialized"
                and "roi_images" in run_group
            ):
                return crop_parent, run_group, candidate

    latest_any = _normalize_run_name(crop_parent.attrs.get("latest_any"))
    if latest_any and latest_any in crop_parent:
        run_group = crop_parent[latest_any]
        if not is_run_selector_eligible(run_group) or not is_run_complete_in_parent(
            crop_parent,
            run_group,
        ):
            run_group = None
    else:
        run_group = None
    if run_group is not None:
        latest_mode = _resolve_storage_mode(run_group)
        if latest_mode != "materialized":
            raise ValueError(
                f"Latest available crop run '{latest_any}' uses {latest_mode!r} storage. "
                "Traditional pipelines require "
                "a materialized crop run with roi_images."
            )

    raise ValueError(
        "No materialized crop run found under crop_runs. Traditional pipelines require crop_runs/<run>/roi_images."
    )


def _normalize_storage_mode(value: object) -> str | None:
    text = _normalize_run_name(value)
    if text in {"materialized", "geometry_only", COMPOSITE_CROP_STORAGE_MODE}:
        return text
    return None


def _resolve_storage_mode(crop_group: zarr.Group) -> str:
    explicit = _normalize_storage_mode(crop_group.attrs.get("crop_storage_mode"))
    if explicit is not None:
        return explicit
    if "roi_images" in crop_group:
        return "materialized"
    return "geometry_only"


def _is_acquisition_crop_video_source(crop_group: zarr.Group) -> bool:
    for attr_name in ("source_pixels", "roi_pixel_provider", "source_type"):
        if _normalize_run_name(crop_group.attrs.get(attr_name)) in {
            SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
            SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME,
        }:
            return True
    # Compatibility only: old acquisition-crop manifests predate the explicit
    # source profiles. New writers must declare one of the profiles above.
    return "source_crop_video_frame_indices" in crop_group and bool(
        crop_group.attrs.get("source_crop_video_path")
    )


def _resolve_roi_shape(
    crop_group: zarr.Group,
    *,
    crop_run_name: str,
) -> tuple[int, int]:
    strict_shape = strict_crop_fixed_roi_shape(
        crop_group,
        run_id=crop_run_name,
    )
    if strict_shape is not None:
        return strict_shape
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    if "roi_images" in crop_group:
        roi_images = crop_group["roi_images"]
        if len(roi_images.shape) < 3:
            raise ValueError("crop_runs/<run>/roi_images must be at least 3D")
        return int(roi_images.shape[1]), int(roi_images.shape[2])
    if "roi_sizes_full" in crop_group:
        return _resolve_fixed_roi_shape_from_wh(
            np.asarray(crop_group["roi_sizes_full"][:])
        )
    if "source_crop_xywh" in crop_group:
        source_crop_xywh = np.asarray(crop_group["source_crop_xywh"][:])
        if source_crop_xywh.ndim != 2 or source_crop_xywh.shape[1] < 4:
            raise ValueError("crop_runs/<run>/source_crop_xywh must have shape [N,4].")
        return _resolve_fixed_roi_shape_from_wh(source_crop_xywh[:, 2:4])
    raise ValueError(
        "Unable to determine ROI size for crop run (need roi_size or roi_images)."
    )


def _resolve_fixed_roi_shape_from_wh(width_height_rows: np.ndarray) -> tuple[int, int]:
    rows = np.asarray(width_height_rows)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError(
            "ROI size rows must have shape [N,2] with width,height columns."
        )
    if rows.shape[0] == 0:
        raise ValueError(
            "Unable to determine ROI size from an empty geometry-only crop run."
        )
    finite = np.isfinite(rows[:, :2]).all(axis=1)
    positive = np.logical_and(rows[:, 0] > 0, rows[:, 1] > 0)
    valid = np.logical_and(finite, positive)
    if not np.any(valid):
        raise ValueError(
            "Unable to determine ROI size: no positive finite width/height rows."
        )
    rounded = np.rint(rows[valid, :2]).astype(np.int64, copy=False)
    unique = np.unique(rounded, axis=0)
    if unique.shape[0] != 1:
        preview = unique[:5].tolist()
        raise ValueError(
            "CropImageSource requires fixed-size ROI rows; found multiple "
            f"width,height values: {preview}"
        )
    width, height = int(unique[0, 0]), int(unique[0, 1])
    return height, width


def _normalize_roi_cache_policy(value: object) -> str:
    text = _normalize_run_name(value) or "never"
    if text not in _ROI_CACHE_POLICIES:
        choices = ", ".join(sorted(_ROI_CACHE_POLICIES))
        raise ValueError(
            f"Invalid roi_cache_policy '{text}'. Expected one of: {choices}"
        )
    return text


def _normalize_roi_live_acceleration(value: object) -> str:
    text = _normalize_run_name(value) or "auto"
    if text not in _ROI_LIVE_ACCELERATION_CHOICES:
        choices = ", ".join(sorted(_ROI_LIVE_ACCELERATION_CHOICES))
        raise ValueError(
            f"Invalid roi_live_acceleration '{text}'. Expected one of: {choices}"
        )
    return text


def _resolve_frame_shape(
    root: zarr.Group,
    crop_group: zarr.Group,
    images_full: object | None,
    *,
    crop_run_name: str,
) -> tuple[int, int] | None:
    strict_shape = strict_crop_source_frame_shape(
        crop_group,
        run_id=crop_run_name,
    )
    if strict_shape is not None:
        return strict_shape
    if images_full is not None:
        shape = getattr(images_full, "shape", ())
        if len(shape) >= 3:
            return int(shape[1]), int(shape[2])

    width = (
        root.attrs.get("video_width")
        or root.attrs.get("width")
        or root.attrs.get("source_video_width")
        or crop_group.attrs.get("width")
    )
    height = (
        root.attrs.get("video_height")
        or root.attrs.get("height")
        or root.attrs.get("source_video_height")
        or crop_group.attrs.get("height")
    )
    if width is None or height is None:
        return None
    return int(height), int(width)


def _resolve_roi_cache_root(roi_cache_dir: str | Path | None) -> Path:
    if roi_cache_dir is not None:
        return Path(roi_cache_dir).expanduser().resolve()

    env_root = os.environ.get("PALETTE_ROI_CACHE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return Path(tempfile.gettempdir()).resolve() / "palette_roi_cache"


def _cache_component(text: str, *, default: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text.strip()
    )
    return cleaned or default


def _cache_runtime_summary(cache_group: zarr.Group) -> str:
    backend = (
        normalize_attr(cache_group.attrs.get("cache_write_backend_effective"))
        or "standard_zarr"
    )
    acceleration = normalize_attr(cache_group.attrs.get("cache_acceleration")) or "cpu"
    fallback_reason = normalize_attr(cache_group.attrs.get("cache_fallback_reason"))
    summary = f"acceleration={acceleration}, backend={backend}"
    if fallback_reason:
        summary += f", fallback={fallback_reason}"
    return summary


def _to_grayscale_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return rgb_to_gray_bt601_cv2_uint8(arr)
    raise ValueError(f"Unsupported frame shape for grayscale conversion: {arr.shape}")


def _normalize_roi_batch(batch: np.ndarray) -> np.ndarray:
    arr = np.asarray(batch)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 3:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 4:
        gray = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.uint8)
        for idx, frame in enumerate(arr):
            gray[idx] = _to_grayscale_uint8(frame)
        return gray
    raise ValueError(f"Unsupported ROI batch shape: {arr.shape}")


def _image_representation_from_contract(contract: Mapping[str, Any] | None) -> str:
    if contract is None:
        return ROI_IMAGE_REPRESENTATION
    return str(contract.get("image_representation") or ROI_IMAGE_REPRESENTATION)


def _resolve_crop_group_pixel_contract(
    crop_group: zarr.Group,
    *,
    crop_storage_mode: str,
    video_source_type: object,
    acceleration: object,
    frame_source_kind: str,
    roi_live_acceleration_effective: str | None,
) -> dict[str, Any]:
    stored = normalize_pixel_contract(
        crop_group.attrs.get("roi_pixel_contract")
    ) or normalize_pixel_contract(crop_group.attrs.get("crop_pixel_contract"))
    pixel_stored_modes = {"materialized", COMPOSITE_CROP_STORAGE_MODE}
    if stored is not None and (
        crop_storage_mode in pixel_stored_modes
        or frame_source_kind == "acquisition_crop_video"
    ):
        return stored
    if crop_storage_mode in pixel_stored_modes:
        return crop_run_pixel_contract(
            crop_storage_mode=crop_storage_mode,
            video_source_type=str(video_source_type or ""),
            acceleration=str(acceleration or ""),
        )
    return crop_image_source_live_pixel_contract(
        frame_source_kind=frame_source_kind,
        roi_live_acceleration_effective=roi_live_acceleration_effective,
    )


def _crop_from_top_left(
    frame: np.ndarray,
    top_left_xy: Sequence[int | np.integer],
    roi_shape: tuple[int, int],
) -> np.ndarray:
    roi_h, roi_w = roi_shape
    x1 = int(top_left_xy[0])
    y1 = int(top_left_xy[1])
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    height, width = frame.shape[:2]
    vy1 = max(0, y1)
    vy2 = min(height, y2)
    vx1 = max(0, x1)
    vx2 = min(width, x2)

    if vy1 >= vy2 or vx1 >= vx2:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    if (
        vy2 - vy1 == roi_h
        and vx2 - vx1 == roi_w
        and 0 <= y1
        and 0 <= x1
        and y2 <= height
        and x2 <= width
    ):
        return frame[vy1:vy2, vx1:vx2].astype(np.uint8, copy=False)

    roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
    py1 = max(0, -y1)
    px1 = max(0, -x1)
    roi[py1 : py1 + vy2 - vy1, px1 : px1 + vx2 - vx1] = frame[vy1:vy2, vx1:vx2]
    return roi


def _check_external_video_live_gpu_available() -> tuple[bool, str]:
    try:
        from fisheye.tracking.crop import check_gpu_crop_available
    except Exception as exc:  # pragma: no cover - defensive import fallback
        return False, f"gpu_crop_import_failed: {exc}"
    return check_gpu_crop_available()


def _read_external_video_live_gpu_batch(
    *,
    video_path: Path,
    frame_indices: np.ndarray,
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    gpu_chunk_frames: int,
) -> np.ndarray:
    frame_indices_np = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    roi_coordinates_np = np.asarray(roi_coordinates_full, dtype=np.int32)
    if roi_coordinates_np.shape[0] != frame_indices_np.shape[0]:
        raise ValueError(
            "frame_indices and roi_coordinates_full must have matching leading dimensions"
        )

    roi_h, roi_w = roi_shape
    batch = np.zeros((frame_indices_np.shape[0], roi_h, roi_w), dtype=np.uint8)
    if frame_indices_np.size == 0:
        return batch

    from fisheye.tracking import crop as tracking_crop

    if (
        tracking_crop.VideoReader is None
        or tracking_crop.gpu is None
        or tracking_crop.decord is None
    ):
        raise RuntimeError("GPU live ROI reads require Decord GPU video support.")
    if not getattr(tracking_crop, "_TORCH_AVAILABLE", False):
        raise RuntimeError("GPU live ROI reads require PyTorch.")

    frame_to_roi: dict[int, list[int]] = {}
    for local_roi_idx, frame_idx in enumerate(frame_indices_np.tolist()):
        frame_to_roi.setdefault(int(frame_idx), []).append(int(local_roi_idx))

    unique_frames = sorted(frame_to_roi.keys())
    chunk_len = max(1, int(gpu_chunk_frames))
    video_reader = None
    try:
        tracking_crop.decord.bridge.set_bridge("torch")
        video_reader = tracking_crop.VideoReader(
            str(video_path), ctx=tracking_crop.gpu(0)
        )
        for chunk_idx, start in enumerate(range(0, len(unique_frames), chunk_len)):
            chunk_frames = unique_frames[start : start + chunk_len]
            if not chunk_frames:
                continue
            frames_gpu = video_reader.get_batch(chunk_frames)
            _, roi_ids, crops_cpu, _coords_cpu, _chunk_time = (
                tracking_crop._process_chunk_gpu_from_top_left(
                    chunk_idx,
                    chunk_frames,
                    frames_gpu,
                    frame_to_roi,
                    roi_coordinates_np,
                    roi_shape,
                    video_shape,
                    return_device=False,
                )
            )
            if roi_ids.size > 0:
                batch[roi_ids] = crops_cpu
            del frames_gpu
    finally:
        try:
            if tracking_crop.decord is not None:
                tracking_crop.decord.bridge.set_bridge("native")
        except Exception:
            pass
        video_reader = None
        torch_mod = getattr(tracking_crop, "torch", None)
        try:
            if torch_mod is not None and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            pass

    return batch


class _ExternalFrameReader:
    """Explicit CPU-only external-video ROI reader for inspection/non-production use."""

    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self._reader = None
        self._backend: str | None = None
        self._capture = None

    @property
    def backend(self) -> str:
        if self._backend is None:
            self._ensure_reader()
        assert self._backend is not None
        return self._backend

    def _ensure_reader(self) -> None:
        if self._reader is not None or self._capture is not None:
            return
        if not self.video_path.exists():
            raise FileNotFoundError(f"Source video not found: {self.video_path}")

        if VideoReader is not None and cpu is not None:
            try:  # pragma: no cover - backend availability depends on runtime
                decord.bridge.set_bridge("native")
                self._reader = VideoReader(str(self.video_path), ctx=cpu())
                self._backend = "decord_cpu"
                return
            except Exception:
                self._reader = None

        if cv2 is not None:
            capture = cv2.VideoCapture(str(self.video_path))
            if capture.isOpened():
                self._capture = capture
                self._backend = "opencv"
                return
            capture.release()

        if _DECORD_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Unable to open source video for live ROI reads. "
                f"Decord import failed: {_DECORD_IMPORT_ERROR}"
            )
        if _CV2_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Unable to open source video for live ROI reads. "
                f"OpenCV import failed: {_CV2_IMPORT_ERROR}"
            )
        raise RuntimeError(
            f"Unable to open source video for live ROI reads: {self.video_path}"
        )

    def read_frame(self, frame_idx: int) -> np.ndarray:
        self._ensure_reader()
        if self._reader is not None:
            frame = self._reader[int(frame_idx)]
            if hasattr(frame, "asnumpy"):
                frame = frame.asnumpy()
            elif hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            return _to_grayscale_uint8(frame)

        assert self._capture is not None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} from {self.video_path}"
            )
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return _to_grayscale_uint8(frame)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._reader = None


class _AcquisitionCropVideoFrameReader:
    """Sequential PyNvVC luma reader for acquisition-time crop videos."""

    def __init__(self, video_path: Path, *, gpu_id: int = 0) -> None:
        self.video_path = video_path
        self.gpu_id = int(gpu_id)
        self._reader = None
        self._frame_iter = None
        self._next_frame_index = 0
        self._source_height: int | None = None
        self._source_width: int | None = None

    @property
    def source_shape(self) -> tuple[int, int] | None:
        if self._source_height is None or self._source_width is None:
            return None
        return self._source_height, self._source_width

    def _ensure_reader(self) -> None:
        if self._reader is not None and self._frame_iter is not None:
            return
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Acquisition crop video not found: {self.video_path}"
            )
        if PynvvcLumaRgbReader is None:
            raise RuntimeError(
                "PyNvVideoCodec luma reader is unavailable; cannot read acquisition crop video "
                f"{self.video_path}: {_PYNVVC_IMPORT_ERROR}"
            )
        self._reader = PynvvcLumaRgbReader(
            self.video_path, start_frame=0, gpu_id=self.gpu_id
        )
        self._frame_iter = iter(self._reader.iter_frames())
        self._next_frame_index = 0
        self._source_height = int(self._reader.source_height)
        self._source_width = int(self._reader.source_width)

    def _reset_reader(self) -> None:
        self.close()
        self._ensure_reader()

    def read_frame(self, frame_idx: int) -> np.ndarray:
        target = int(frame_idx)
        if target < 0:
            raise IndexError(f"Negative acquisition crop-video frame index: {target}")
        if target < self._next_frame_index:
            self._reset_reader()
        else:
            self._ensure_reader()

        assert self._frame_iter is not None
        assert self._source_height is not None
        assert self._source_width is not None
        while self._next_frame_index <= target:
            try:
                frame = next(self._frame_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Acquisition crop video ended before frame {target} could be decoded: {self.video_path}"
                ) from exc
            current = self._next_frame_index
            self._next_frame_index += 1
            if current == target:
                luma = frame[: self._source_height, : self._source_width]
                if hasattr(luma, "contiguous"):
                    luma = luma.contiguous()
                if hasattr(luma, "to"):
                    luma = luma.to("cpu")
                if hasattr(luma, "numpy"):
                    luma = luma.numpy()
                return np.asarray(luma, dtype=np.uint8).copy()
        raise RuntimeError(f"Failed to decode acquisition crop-video frame {target}")

    def close(self) -> None:
        if self._reader is not None and hasattr(self._reader, "close"):
            self._reader.close()
        self._reader = None
        self._frame_iter = None
        self._next_frame_index = 0
        self._source_height = None
        self._source_width = None


def _collection_crop_video_readers(
    crop_group: zarr.Group,
) -> dict[int, _AcquisitionCropVideoFrameReader]:
    raw_members = crop_group.attrs.get("source_media_members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError(
            "Acquisition crop-video collection lacks source_media_members provenance."
        )
    readers: dict[int, _AcquisitionCropVideoFrameReader] = {}
    for expected_index, raw in enumerate(raw_members):
        if not isinstance(raw, Mapping):
            raise ValueError("Crop-video collection member must be an object.")
        member_index = int(raw.get("member_index", -1))
        if member_index != expected_index:
            raise ValueError("Crop-video collection member indices are not contiguous.")
        media = raw.get("crop_video")
        if not isinstance(media, Mapping):
            raise ValueError(
                f"Crop-video collection member {member_index} lacks crop media."
            )
        path_text = str(media.get("path") or media.get("resolved_path") or "").strip()
        if not path_text:
            raise ValueError(
                f"Crop-video collection member {member_index} lacks a media path."
            )
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Acquisition crop-video collection member not found: {path}"
            )
        try:
            expected_size = int(media.get("size_bytes"))
            expected_mtime = int(media.get("mtime_ns"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Crop-video collection member {member_index} lacks stat identity."
            ) from exc
        stat = path.stat()
        if stat.st_size != expected_size or stat.st_mtime_ns != expected_mtime:
            raise ValueError(
                f"Crop-video collection member {member_index} changed after publication."
            )
        readers[member_index] = _AcquisitionCropVideoFrameReader(path)
    return readers


@dataclass
class CropImageSource:
    """Read ROI pixels from a crop run regardless of storage mode."""

    root: zarr.Group
    crop_group: zarr.Group
    crop_run_name: str
    storage_mode: str
    roi_shape: tuple[int, int]
    roi_coordinates_full: np.ndarray
    frame_indices: np.ndarray
    frame_source_kind: str
    frame_source_path: str | None
    frame_source_declared_path: str | None = None
    frame_source_path_override_used: bool = False
    source_video_frame_offset: int = 0
    source_video_frame_count: int | None = None
    frame_shape: tuple[int, int] | None = None
    roi_read_mode: str = "materialized_crop_run"
    roi_cache_policy: str = "never"
    roi_cache_used: bool = False
    roi_cache_created: bool = False
    roi_cache_key: str | None = None
    roi_cache_path: str | None = None
    roi_cache_canonical_path: str | None = None
    roi_cache_backend: str | None = None
    roi_image_representation: str | None = ROI_IMAGE_REPRESENTATION
    roi_pixel_contract: dict[str, Any] | None = None
    roi_live_acceleration_requested: str | None = None
    roi_live_acceleration_effective: str | None = None
    roi_live_acceleration_fallback_reason: str | None = None
    roi_live_gpu_chunk_frames: int = _ROI_LIVE_GPU_CHUNK_FRAMES_DEFAULT
    source_crop_row_ids: np.ndarray | None = None
    pixel_materialization_id: str | None = None
    pixel_materialization_manifest: str | None = None
    pixel_source_crop_run_name: str | None = None
    geometry_crop_rebase: dict[str, Any] | None = None
    _roi_images: object | None = None
    _images_full: object | None = None
    _external_reader: _ExternalFrameReader | None = None
    crop_video_frame_indices: np.ndarray | None = None
    crop_video_member_indices: np.ndarray | None = None
    source_pixel_kind_codes: np.ndarray | None = None
    supplemental_cache_row_indices: np.ndarray | None = None
    _acquisition_crop_reader: _AcquisitionCropVideoFrameReader | None = None
    _acquisition_crop_readers: dict[int, _AcquisitionCropVideoFrameReader] | None = None
    _supplemental_flat_cache: object | None = None

    @classmethod
    def open(
        cls,
        root: zarr.Group,
        *,
        crop_run: str | None = None,
        zarr_path: str | Path | None = None,
        roi_cache_policy: str = "never",
        roi_live_acceleration: str = "auto",
        roi_live_gpu_chunk_frames: int = _ROI_LIVE_GPU_CHUNK_FRAMES_DEFAULT,
        roi_cache_dir: str | Path | None = None,
        roi_cache_manifest: str | Path | None = None,
        roi_cache_expected_archive_path: str | Path | None = None,
        source_video_path_override: str | Path | None = None,
        source_video_frame_offset: int = 0,
        source_video_frame_count: int | None = None,
        source_crop_row_start: int | None = None,
        source_crop_row_stop: int | None = None,
        console: Any | None = None,
    ) -> "CropImageSource":
        normalized_cache_policy = _normalize_roi_cache_policy(roi_cache_policy)
        normalized_live_acceleration = _normalize_roi_live_acceleration(
            roi_live_acceleration
        )
        live_gpu_chunk_frames = max(1, int(roi_live_gpu_chunk_frames))
        manifest_path = (
            Path(roi_cache_manifest).expanduser()
            if roi_cache_manifest is not None
            else None
        )
        video_override = (
            Path(source_video_path_override).expanduser().resolve()
            if source_video_path_override is not None
            else None
        )
        video_frame_offset = int(source_video_frame_offset)
        video_frame_count = (
            int(source_video_frame_count)
            if source_video_frame_count is not None
            else None
        )
        if video_frame_offset < 0:
            raise ValueError("source_video_frame_offset must be nonnegative.")
        if video_frame_count is not None and video_frame_count <= 0:
            raise ValueError("source_video_frame_count must be positive when provided.")
        if video_frame_offset != 0 and video_frame_count is None:
            raise ValueError(
                "A nonzero source_video_frame_offset requires source_video_frame_count."
            )
        if (
            video_frame_offset != 0 or video_frame_count is not None
        ) and video_override is None:
            raise ValueError(
                "A source-video frame window requires source_video_path_override."
            )
        if video_override is not None and not video_override.is_file():
            raise FileNotFoundError(
                f"Source-video path override does not exist: {video_override}"
            )
        if (source_crop_row_start is None) != (source_crop_row_stop is None):
            raise ValueError(
                "source_crop_row_start and source_crop_row_stop must be provided together."
            )
        if crop_run is None and manifest_path is not None:
            crop_run = crop_run_name_from_manifest(manifest_path)
        crop_parent, crop_group, crop_run_name = resolve_crop_run(
            root,
            crop_run=crop_run,
            zarr_path=zarr_path,
        )
        storage_mode = _resolve_storage_mode(crop_group)
        roi_shape = _resolve_roi_shape(
            crop_group,
            crop_run_name=crop_run_name,
        )

        if "roi_coordinates_full" not in crop_group:
            raise ValueError("Crop run missing 'roi_coordinates_full'.")
        roi_coordinates_full = np.asarray(
            crop_group["roi_coordinates_full"][:], dtype=np.int32
        )
        total_rois = int(roi_coordinates_full.shape[0])

        frame_indices_arr = crop_group.get("frame_indices")
        if frame_indices_arr is not None:
            frame_indices = np.asarray(frame_indices_arr[:], dtype=np.int64)
        elif storage_mode == "materialized":
            frame_indices = np.zeros(total_rois, dtype=np.int64)
        else:
            raise ValueError(f"{storage_mode!r} crop run is missing 'frame_indices'.")

        if frame_indices.shape[0] != total_rois:
            raise ValueError(
                f"frame_indices length {frame_indices.shape[0]} does not match roi_coordinates_full length {total_rois}"
            )

        roi_images = crop_group.get("roi_images")
        crop_video_frame_indices = None
        crop_video_member_indices = None
        source_pixel_kind_codes = None
        supplemental_cache_row_indices = None
        supplemental_flat_cache = None
        acquisition_crop_reader = None
        acquisition_crop_readers = None
        if storage_mode == "materialized":
            if video_override is not None:
                raise ValueError(
                    "source_video_path_override is valid only for geometry-only "
                    "raw-camera-video crop runs."
                )
            if roi_images is None:
                raise ValueError("Materialized crop run is missing 'roi_images'.")
            frame_source_kind = "roi_images"
            frame_source_path = None
            images_full = None
            external_reader = None
            frame_shape = roi_shape
            roi_read_mode = "materialized_crop_run"
            live_acceleration_effective = None
            live_acceleration_fallback_reason = None
        elif storage_mode == COMPOSITE_CROP_STORAGE_MODE:
            if video_override is not None:
                raise ValueError(
                    "source_video_path_override is valid only for geometry-only "
                    "raw-camera-video crop runs."
                )
            roi_images = CompositeCropArray.open(
                crop_parent,
                crop_group,
                run_name=crop_run_name,
                verify_identity=False,
            )
            frame_source_kind = "composite_crop"
            frame_source_path = None
            images_full = None
            external_reader = None
            frame_shape = roi_shape
            roi_read_mode = "composite_base_delta"
            live_acceleration_effective = None
            live_acceleration_fallback_reason = None
        else:
            raw_video = root.get("raw_video")
            images_full = (
                raw_video.get("images_full") if raw_video is not None else None
            )
            acquisition_crop_video = _is_acquisition_crop_video_source(crop_group)
            if acquisition_crop_video:
                if video_override is not None:
                    raise ValueError(
                        "source_video_path_override cannot replace an acquisition "
                        "crop-video authority."
                    )
                crop_video_frame_indices_arr = crop_group.get(
                    "source_crop_video_frame_indices"
                )
                if crop_video_frame_indices_arr is None:
                    raise ValueError(
                        "Acquisition crop-video crop run is missing source_crop_video_frame_indices."
                    )
                crop_video_frame_indices = np.asarray(
                    crop_video_frame_indices_arr[:], dtype=np.int64
                )
                if crop_video_frame_indices.shape[0] != total_rois:
                    raise ValueError(
                        "source_crop_video_frame_indices length "
                        f"{crop_video_frame_indices.shape[0]} does not match roi count {total_rois}"
                    )
                crop_video_member_indices_arr = crop_group.get(
                    "source_crop_video_member_indices"
                )
                if crop_video_member_indices_arr is not None:
                    crop_video_member_indices = np.asarray(
                        crop_video_member_indices_arr[:], dtype=np.int32
                    )
                    if crop_video_member_indices.shape[0] != total_rois:
                        raise ValueError(
                            "source_crop_video_member_indices length "
                            f"{crop_video_member_indices.shape[0]} does not match roi count {total_rois}"
                        )
                    acquisition_crop_readers = _collection_crop_video_readers(
                        crop_group
                    )
                    invalid_members = np.logical_and(
                        crop_video_member_indices >= 0,
                        ~np.isin(
                            crop_video_member_indices,
                            np.asarray(
                                sorted(acquisition_crop_readers), dtype=np.int32
                            ),
                        ),
                    )
                    if np.any(invalid_members):
                        raise ValueError(
                            "source_crop_video_member_indices references an unknown collection member."
                        )
                    crop_video_path = None
                else:
                    crop_video_path = (
                        crop_group.attrs.get("source_crop_video_path")
                        or crop_group.attrs.get("source_video_path")
                        or crop_group.attrs.get("video_source_path")
                    )
                    if not crop_video_path:
                        raise ValueError(
                            "Acquisition crop-video crop run requires source_crop_video_path provenance."
                        )
                source_pixel_kind_codes_arr = crop_group.get("source_pixel_kind_codes")
                if source_pixel_kind_codes_arr is not None:
                    source_pixel_kind_codes = np.asarray(
                        source_pixel_kind_codes_arr[:],
                        dtype=np.int16,
                    )
                    if source_pixel_kind_codes.shape[0] != total_rois:
                        raise ValueError(
                            "source_pixel_kind_codes length "
                            f"{source_pixel_kind_codes.shape[0]} does not match roi count {total_rois}"
                        )
                    declared_pixel_source = str(
                        crop_group.attrs.get("source_pixels")
                        or crop_group.attrs.get("roi_pixel_provider")
                        or ""
                    ).strip()
                    if (
                        declared_pixel_source
                        == SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME
                        and np.any(~np.isin(source_pixel_kind_codes, (0, 1)))
                    ):
                        raise ValueError(
                            "Hybrid acquisition crop run source_pixel_kind_codes "
                            "must contain only 0 (acquisition crop video) or 1 "
                            "(full-frame supplemental cache)."
                        )
                    if np.any(source_pixel_kind_codes != 0):
                        supplemental_cache_rows_arr = crop_group.get(
                            "supplemental_cache_row_indices"
                        )
                        if supplemental_cache_rows_arr is None:
                            raise ValueError(
                                "Hybrid acquisition crop run has non-video source rows but is "
                                "missing supplemental_cache_row_indices."
                            )
                        supplemental_cache_row_indices = np.asarray(
                            supplemental_cache_rows_arr[:],
                            dtype=np.int64,
                        )
                        if supplemental_cache_row_indices.shape[0] != total_rois:
                            raise ValueError(
                                "supplemental_cache_row_indices length "
                                f"{supplemental_cache_row_indices.shape[0]} does not match roi count {total_rois}"
                            )
                        supplemental_manifest = crop_group.attrs.get(
                            "supplemental_roi_cache_manifest"
                        ) or crop_group.attrs.get(
                            "supplemental_flat_roi_cache_manifest"
                        )
                        if not supplemental_manifest:
                            raise ValueError(
                                "Hybrid acquisition crop run has supplemental rows but is "
                                "missing supplemental_roi_cache_manifest provenance."
                            )
                        supplemental_flat_cache = open_flat_roi_cache(
                            Path(str(supplemental_manifest)).expanduser(),
                            expected_crop_run=crop_run_name,
                        )
                        if (
                            declared_pixel_source
                            == SOURCE_PIXELS_HYBRID_ACQUISITION_FULL_FRAME
                        ):
                            builder = supplemental_flat_cache.manifest.get("builder")
                            supplemental_contract = (
                                normalize_pixel_contract(builder.get("pixel_contract"))
                                if isinstance(builder, Mapping)
                                else None
                            )
                            expected_supplemental_contract = (
                                orange_mono_pynvvc_luma_pixel_contract(
                                    source_pixels=SOURCE_PIXELS_RAW_CAMERA_VIDEO,
                                )
                            )
                            if supplemental_contract != expected_supplemental_contract:
                                supplemental_flat_cache.close()
                                raise ValueError(
                                    "Hybrid acquisition crop run supplemental cache "
                                    "does not carry the authoritative raw-camera "
                                    "PyNvVC luma pixel contract."
                                )
                frame_source_kind = "acquisition_crop_video"
                frame_source_path = (
                    str(crop_video_path) if crop_video_path is not None else None
                )
                frame_shape = roi_shape
                images_full = None
                external_reader = None
                if crop_video_path is not None:
                    acquisition_crop_reader = _AcquisitionCropVideoFrameReader(
                        Path(str(crop_video_path))
                    )
                live_acceleration_effective = "pynvvc_luma"
                live_acceleration_fallback_reason = None
            elif manifest_path is not None:
                if video_override is not None:
                    raise ValueError(
                        "source_video_path_override and roi_cache_manifest are "
                        "mutually exclusive."
                    )
                frame_shape = _resolve_frame_shape(
                    root,
                    crop_group,
                    images_full,
                    crop_run_name=crop_run_name,
                )
                frame_source_kind = "flat_roi_cache_manifest"
                frame_source_path = str(manifest_path)
                images_full = None
                external_reader = None
                live_acceleration_effective = None
                live_acceleration_fallback_reason = None
            else:
                frame_shape = _resolve_frame_shape(
                    root,
                    crop_group,
                    images_full,
                    crop_run_name=crop_run_name,
                )
                if images_full is not None:
                    if normalized_live_acceleration == "gpu":
                        raise ValueError(
                            "roi_live_acceleration='gpu' is only supported for geometry-only external-video reads."
                        )
                    frame_source_kind = "raw_video/images_full"
                    frame_source_path = None
                    external_reader = None
                    live_acceleration_effective = "cpu"
                    live_acceleration_fallback_reason = None
                else:
                    source_video_path = crop_group.attrs.get(
                        "source_video_path"
                    ) or crop_group.attrs.get("video_source_path")
                    if not source_video_path:
                        try:
                            source_video_path = str(
                                resolve_source_video(
                                    root,
                                    zarr_path=(
                                        Path(zarr_path)
                                        if zarr_path is not None
                                        else None
                                    ),
                                ).path
                            )
                        except SourceVideoMetadataMissingError:
                            source_video_path = None
                    if not source_video_path:
                        raise ValueError(
                            "Geometry-only crop run requires raw_video/images_full or source_video_path provenance."
                        )
                    declared_source_video_path = str(source_video_path)
                    frame_source_kind = "source_video_path"
                    frame_source_path = str(
                        video_override
                        if video_override is not None
                        else declared_source_video_path
                    )
                    if video_override is not None:
                        metadata = root.attrs.get("source_video_metadata")
                        fingerprint = (
                            metadata.get("file_fingerprint")
                            if isinstance(metadata, Mapping)
                            else None
                        )
                        expected_size = (
                            fingerprint.get("size_bytes")
                            if isinstance(fingerprint, Mapping)
                            else None
                        )
                        if (
                            expected_size is not None
                            and video_frame_count is None
                            and int(video_override.stat().st_size) != int(expected_size)
                        ):
                            raise ValueError(
                                "Source-video path override size differs from the "
                                "declared source-video fingerprint."
                            )
                    external_reader = _ExternalFrameReader(Path(frame_source_path))
                    if normalized_live_acceleration == "cpu":
                        live_acceleration_effective = "cpu"
                        live_acceleration_fallback_reason = None
                    else:
                        gpu_available, gpu_reason = (
                            _check_external_video_live_gpu_available()
                        )
                        if normalized_live_acceleration == "gpu":
                            if not gpu_available:
                                raise _gpu_decode_unavailable(
                                    f"roi_live_acceleration='gpu' requested but unavailable: {gpu_reason}"
                                )
                            if frame_shape is None:
                                raise ValueError(
                                    "roi_live_acceleration='gpu' requires known source frame dimensions "
                                    "(video_width/video_height metadata)."
                                )
                            live_acceleration_effective = "gpu"
                            live_acceleration_fallback_reason = None
                        elif gpu_available and frame_shape is not None:
                            live_acceleration_effective = "gpu"
                            live_acceleration_fallback_reason = None
                        else:
                            if frame_shape is None:
                                reason = "unknown_frame_shape"
                            else:
                                reason = gpu_reason
                            raise _gpu_decode_unavailable(reason)
            if frame_source_kind == "acquisition_crop_video":
                roi_read_mode = "acquisition_crop_video"
            elif frame_source_kind == "flat_roi_cache_manifest":
                roi_read_mode = "flat_roi_cache_manifest"
            else:
                roi_read_mode = "geometry_only_live"

        roi_pixel_contract = _resolve_crop_group_pixel_contract(
            crop_group,
            crop_storage_mode=storage_mode,
            video_source_type=crop_group.attrs.get("video_source_type"),
            acceleration=crop_group.attrs.get("acceleration"),
            frame_source_kind=frame_source_kind,
            roi_live_acceleration_effective=live_acceleration_effective,
        )
        source = cls(
            root=root,
            crop_group=crop_group,
            crop_run_name=crop_run_name,
            storage_mode=storage_mode,
            roi_shape=roi_shape,
            roi_coordinates_full=roi_coordinates_full,
            frame_indices=frame_indices,
            frame_source_kind=frame_source_kind,
            frame_source_path=frame_source_path,
            frame_source_declared_path=(
                declared_source_video_path
                if storage_mode == "geometry_only"
                and frame_source_kind == "source_video_path"
                else None
            ),
            frame_source_path_override_used=bool(
                video_override is not None
                and storage_mode == "geometry_only"
                and frame_source_kind == "source_video_path"
            ),
            source_video_frame_offset=(
                video_frame_offset if frame_source_kind == "source_video_path" else 0
            ),
            source_video_frame_count=(
                video_frame_count if frame_source_kind == "source_video_path" else None
            ),
            frame_shape=frame_shape,
            roi_read_mode=roi_read_mode,
            roi_cache_policy=normalized_cache_policy,
            roi_image_representation=_image_representation_from_contract(
                roi_pixel_contract
            ),
            roi_pixel_contract=roi_pixel_contract,
            roi_live_acceleration_requested=(
                normalized_live_acceleration
                if storage_mode == "geometry_only"
                else None
            ),
            roi_live_acceleration_effective=(
                live_acceleration_effective if storage_mode == "geometry_only" else None
            ),
            roi_live_acceleration_fallback_reason=(
                live_acceleration_fallback_reason
                if storage_mode == "geometry_only"
                else None
            ),
            roi_live_gpu_chunk_frames=live_gpu_chunk_frames,
            _roi_images=roi_images,
            _images_full=images_full if storage_mode == "geometry_only" else None,
            _external_reader=(
                external_reader if storage_mode == "geometry_only" else None
            ),
            crop_video_frame_indices=crop_video_frame_indices,
            crop_video_member_indices=crop_video_member_indices,
            source_pixel_kind_codes=source_pixel_kind_codes,
            supplemental_cache_row_indices=supplemental_cache_row_indices,
            _acquisition_crop_reader=(
                acquisition_crop_reader if storage_mode == "geometry_only" else None
            ),
            _acquisition_crop_readers=(
                acquisition_crop_readers if storage_mode == "geometry_only" else None
            ),
            _supplemental_flat_cache=(
                supplemental_flat_cache if storage_mode == "geometry_only" else None
            ),
        )
        if source_crop_row_start is not None and source_crop_row_stop is not None:
            start = int(source_crop_row_start)
            stop = int(source_crop_row_stop)
            if start < 0 or stop <= start or stop > source.total_rois:
                raise ValueError(
                    f"Invalid source crop-row interval [{start}, {stop}) for "
                    f"{source.total_rois} rows."
                )
            source._select_source_crop_rows(np.arange(start, stop, dtype=np.int64))
        if manifest_path is not None:
            source._activate_flat_bin_cache(
                manifest_path=manifest_path,
                zarr_path=(
                    roi_cache_expected_archive_path
                    if roi_cache_expected_archive_path is not None
                    else zarr_path
                ),
            )
        elif source.storage_mode == "geometry_only" and source._should_use_roi_cache():
            source._activate_temporary_cache(
                zarr_path=zarr_path,
                roi_cache_dir=roi_cache_dir,
                console=console,
            )
        return source

    def _select_source_crop_rows(self, rows: np.ndarray) -> None:
        """Restrict a live source to an ordered global crop-row partition."""

        selected = np.asarray(rows, dtype=np.int64).reshape(-1)
        original_rows = self.total_rois
        if selected.size == 0 or selected.min() < 0 or selected.max() >= original_rows:
            raise ValueError("Selected source crop rows are empty or out of bounds.")
        if not np.array_equal(selected, np.sort(np.unique(selected))):
            raise ValueError("Selected source crop rows must be unique and ascending.")
        if self._roi_images is not None:
            raise ValueError(
                "Direct crop-row interval selection is supported only for live "
                "geometry-only providers."
            )
        self.roi_coordinates_full = np.asarray(
            self.roi_coordinates_full[selected], dtype=np.int32
        )
        self.frame_indices = np.asarray(self.frame_indices[selected], dtype=np.int64)
        for name in (
            "crop_video_frame_indices",
            "crop_video_member_indices",
            "source_pixel_kind_codes",
            "supplemental_cache_row_indices",
        ):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, np.asarray(values[selected]))
        self.source_crop_row_ids = selected.copy()
        self.roi_read_mode = f"{self.roi_read_mode}_crop_row_partition"

    @classmethod
    def open_work_package(
        cls,
        root: zarr.Group,
        *,
        manifest_path: str | Path,
        zarr_path: str | Path | None = None,
        crop_run: str | None = None,
        verify_payload: bool = True,
        verify_pixel_rows: bool = True,
    ) -> "CropImageSource":
        """Open a keyed subset package while retaining its logical crop binding."""

        from fisheye.shared.crop_pixel_work_package import (
            open_crop_pixel_work_package,
        )

        archive_path = (
            Path(zarr_path).expanduser().resolve() if zarr_path is not None else None
        )
        package = open_crop_pixel_work_package(
            manifest_path,
            expected_archive_path=archive_path,
            expected_crop_run=crop_run,
            root=root,
            verify_payload=verify_payload,
            verify_pixel_rows=verify_pixel_rows,
        )
        try:
            crop_parent, crop_group, crop_run_name = resolve_crop_run(
                root,
                crop_run=package.crop_run_name,
                zarr_path=archive_path,
            )
            del crop_parent
            pixel_contract = dict(package.pixel_contract)
            return cls(
                root=root,
                crop_group=crop_group,
                crop_run_name=crop_run_name,
                storage_mode=_resolve_storage_mode(crop_group),
                roi_shape=package.roi_shape,
                roi_coordinates_full=np.asarray(
                    package.roi_coordinates_full, dtype=np.int32
                ),
                frame_indices=np.asarray(package.frame_indices, dtype=np.int64),
                frame_source_kind="crop_pixel_work_package",
                frame_source_path=None,
                frame_shape=package.roi_shape,
                roi_read_mode="crop_pixel_work_package",
                roi_cache_policy="never",
                roi_cache_used=True,
                roi_cache_created=False,
                roi_cache_key=package.package_id,
                roi_cache_path=str(package.manifest_path),
                roi_cache_canonical_path=str(package.manifest_path),
                roi_cache_backend="keyed_flat_bin_v1",
                roi_image_representation=_image_representation_from_contract(
                    pixel_contract
                ),
                roi_pixel_contract=pixel_contract,
                source_crop_row_ids=np.asarray(
                    package.crop_row_indices, dtype=np.int64
                ),
                pixel_materialization_id=package.package_id,
                pixel_materialization_manifest=str(package.manifest_path),
                _roi_images=package.pixels,
            )
        except Exception:
            package.close()
            raise

    def bind_geometry_crop(
        self,
        geometry_crop_run: str,
        *,
        zarr_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Rebind authenticated pixels to one exact strict crop-v2 rowset.

        Pixel bytes remain owned by the already-open source (flat cache or work
        package).  Only the logical crop identity and placement are rebound, and
        only after an exact instance-keyed comparison proves that every consumed
        pixel row represents the same observation and crop window.
        """

        requested = _normalize_run_name(geometry_crop_run)
        if requested is None:
            raise ValueError("geometry_crop_run must be one nonempty run name.")
        if self.geometry_crop_rebase is not None:
            if requested != self.crop_run_name:
                raise ValueError(
                    "CropImageSource is already bound to a different geometry crop."
                )
            return dict(self.geometry_crop_rebase)
        if requested == self.crop_run_name:
            raise ValueError(
                "Geometry rebase requires distinct pixel-source and geometry crop runs."
            )

        source_group = self.crop_group
        source_run_name = self.crop_run_name
        crop_parent, target_group, target_run_name = resolve_crop_run(
            self.root,
            crop_run=requested,
            zarr_path=zarr_path,
        )
        if not is_run_complete_in_parent(
            crop_parent,
            target_group,
            legacy_default=False,
        ):
            raise ValueError(
                f"Geometry crop {target_run_name!r} is not strictly complete."
            )
        manifest = target_group.attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE)
        if not isinstance(manifest, Mapping):
            raise ValueError(
                f"Geometry crop {target_run_name!r} lacks its strict run_manifest."
            )
        manifest_errors = validate_crop_run_manifest(manifest)
        if manifest_errors:
            raise ValueError(
                f"Geometry crop {target_run_name!r} has an invalid run_manifest: "
                + "; ".join(manifest_errors)
            )
        if manifest.get("schema_version") != (
            CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                "Geometry rebase requires the coordinate-aware crop run manifest v2."
            )
        payload = manifest.get("payload")
        if not isinstance(payload, Mapping) or payload.get("run_id") != target_run_name:
            raise ValueError(
                "Geometry crop group name differs from its run_manifest run_id."
            )

        target_roi_shape = _resolve_roi_shape(
            target_group,
            crop_run_name=target_run_name,
        )
        if tuple(target_roi_shape) != tuple(self.roi_shape):
            raise ValueError(
                "Pixel-source and geometry crop ROI extents differ: "
                f"{self.roi_shape!r} != {target_roi_shape!r}."
            )

        source_row_count = int(source_group["frame_indices"].shape[0])
        if self.source_crop_row_ids is None:
            source_rows = np.arange(source_row_count, dtype=np.int64)
        else:
            source_rows = np.asarray(
                self.source_crop_row_ids,
                dtype=np.int64,
            ).reshape(-1)
        if source_rows.shape != (self.total_rois,):
            raise ValueError(
                "Pixel-source crop-row selection does not match its pixel row count."
            )
        if (
            source_rows.size == 0
            or int(source_rows.min()) < 0
            or int(source_rows.max()) >= source_row_count
            or np.unique(source_rows).size != source_rows.size
        ):
            raise ValueError(
                "Pixel-source crop rows must be one nonempty unique in-range selection."
            )

        required_target_arrays = {
            "instance_key": (np.dtype(np.uint64), ()),
            "frame_indices": (np.dtype(np.int64), ()),
            "source_acquisition_frame_index": (np.dtype(np.int64), ()),
            "roi_coordinates_full": (np.dtype(np.int32), (2,)),
            "source_crop_xywh": (np.dtype(np.float32), (4,)),
        }
        target_row_count: int | None = None
        for name, (expected_dtype, trailing_shape) in required_target_arrays.items():
            if name not in target_group:
                raise ValueError(
                    f"Geometry crop {target_run_name!r} lacks required array {name!r}."
                )
            array = target_group[name]
            shape = tuple(int(value) for value in array.shape)
            if len(shape) != 1 + len(trailing_shape) or shape[1:] != trailing_shape:
                raise ValueError(
                    f"Geometry crop array {name!r} has invalid shape {shape!r}."
                )
            if target_row_count is None:
                target_row_count = shape[0]
            elif shape[0] != target_row_count:
                raise ValueError("Geometry crop arrays have inconsistent row counts.")
            if np.dtype(array.dtype) != expected_dtype:
                raise ValueError(
                    f"Geometry crop array {name!r} has dtype {array.dtype}, "
                    f"expected {expected_dtype}."
                )
        assert target_row_count is not None

        if "instance_key" not in source_group:
            raise ValueError("Pixel-source crop lacks required instance_key identity.")
        source_keys = np.asarray(
            source_group["instance_key"][source_rows],
            dtype=np.uint64,
        ).reshape(-1)
        target_keys = np.asarray(
            target_group["instance_key"][:],
            dtype=np.uint64,
        ).reshape(-1)
        if np.unique(source_keys).size != source_keys.size:
            raise ValueError("Pixel-source instance_key values are not unique.")
        if np.unique(target_keys).size != target_keys.size:
            raise ValueError("Geometry-crop instance_key values are not unique.")
        target_order = np.argsort(target_keys, kind="stable")
        sorted_target_keys = target_keys[target_order]
        positions = np.searchsorted(sorted_target_keys, source_keys)
        if np.any(positions >= sorted_target_keys.size) or not np.array_equal(
            sorted_target_keys[positions], source_keys
        ):
            raise ValueError(
                "Geometry crop does not contain every consumed pixel-source instance_key."
            )
        target_rows = np.asarray(target_order[positions], dtype=np.int64)

        active_frames = np.asarray(self.frame_indices, dtype=np.int64).reshape(-1)
        source_frames = np.asarray(
            source_group["frame_indices"][source_rows],
            dtype=np.int64,
        ).reshape(-1)
        active_origins = np.asarray(self.roi_coordinates_full, dtype=np.int32)
        source_origins = np.asarray(
            source_group["roi_coordinates_full"][source_rows],
            dtype=np.int32,
        )
        if not np.array_equal(active_frames, source_frames) or not np.array_equal(
            active_origins,
            source_origins,
        ):
            raise ValueError(
                "Active pixel materialization rows differ from their source crop binding."
            )

        exact_arrays = (
            "frame_indices",
            "source_acquisition_frame_index",
            "roi_coordinates_full",
            "source_refined_row_ids",
            "source_detect_row_index",
            "source_clip_indices",
            "source_clip_local_frame_indices",
        )
        compared_arrays: list[str] = []
        for name in exact_arrays:
            if name not in source_group or name not in target_group:
                continue
            source_values = np.asarray(source_group[name][source_rows])
            target_values = np.asarray(target_group[name][target_rows])
            if source_values.shape != target_values.shape or not np.array_equal(
                source_values,
                target_values,
            ):
                raise ValueError(
                    f"Geometry crop differs from pixel-source rows for {name!r}."
                )
            compared_arrays.append(name)

        if "source_crop_xywh" not in source_group:
            raise ValueError("Pixel-source crop lacks source_crop_xywh placement.")
        source_xywh = np.asarray(source_group["source_crop_xywh"][source_rows])
        if (
            source_xywh.shape != (self.total_rois, 4)
            or not np.isfinite(source_xywh).all()
        ):
            raise ValueError("Pixel-source source_crop_xywh is not finite [N,4].")
        normalized_source_xywh = source_xywh.astype(np.float32, copy=False)
        target_xywh = np.asarray(target_group["source_crop_xywh"][target_rows])
        if not np.array_equal(normalized_source_xywh, target_xywh):
            raise ValueError(
                "Geometry crop source_crop_xywh differs after strict float32 normalization."
            )

        target_origins = np.asarray(
            target_group["roi_coordinates_full"][target_rows],
            dtype=np.int32,
        )
        target_frames = np.asarray(
            target_group["frame_indices"][target_rows],
            dtype=np.int64,
        )
        if not np.array_equal(target_origins, target_xywh[:, :2].astype(np.int32)):
            raise ValueError(
                "Geometry crop ROI origins differ from its strict source_crop_xywh."
            )

        target_reference = build_crop_run_reference(
            target_group,
            run_id=target_run_name,
        )
        row_mapping = np.ascontiguousarray(
            np.column_stack((source_rows, target_rows)).astype("<i8", copy=False)
        )
        evidence: dict[str, Any] = {
            "schema_id": "palette.crop_pixel_geometry_rebase",
            "schema_version": 1,
            "operation": "exact_instance_key_subset_rebind_v1",
            "pixel_source_crop_run": source_run_name,
            "geometry_crop_run": target_run_name,
            "geometry_crop_reference": target_reference,
            "row_count": int(self.total_rois),
            "row_mapping_sha256": hashlib.sha256(
                row_mapping.view(np.uint8)
            ).hexdigest(),
            "instance_key_sha256": hashlib.sha256(
                np.ascontiguousarray(source_keys.astype("<u8", copy=False)).view(
                    np.uint8
                )
            ).hexdigest(),
            "compared_arrays": compared_arrays,
            "source_crop_xywh_normalization": "finite_values_cast_to_float32_exact_target_match_v1",
            "source_row_signature_comparison": "intentionally_omitted_different_publication_context",
            "validation": {
                "target_strictly_complete": True,
                "target_coordinate_manifest_v2_valid": True,
                "instance_keys_unique_and_complete": True,
                "active_pixel_binding_matches_source": True,
                "row_identity_and_placement_match": True,
            },
        }

        self.pixel_source_crop_run_name = source_run_name
        self.crop_group = target_group
        self.crop_run_name = target_run_name
        self.storage_mode = _resolve_storage_mode(target_group)
        self.roi_shape = target_roi_shape
        self.roi_coordinates_full = np.ascontiguousarray(target_origins)
        self.frame_indices = np.ascontiguousarray(target_frames)
        self.source_crop_row_ids = np.ascontiguousarray(target_rows)
        self.geometry_crop_rebase = evidence
        return dict(evidence)

    @property
    def total_rois(self) -> int:
        return int(self.roi_coordinates_full.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.total_rois), int(self.roi_shape[0]), int(self.roi_shape[1]))

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    @property
    def roi_array(self) -> object | None:
        """Return the active ROI array backing this source when one exists."""
        return self._roi_images

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.total_rois)
            if step != 1:
                indices = np.arange(start, stop, step, dtype=np.int64)
                return self.read_indices(indices)
            return self.read_slice(start, stop)
        if isinstance(key, (list, tuple, np.ndarray)):
            return self.read_indices(key)
        index = int(key)
        if index < 0:
            index += self.total_rois
        if index < 0 or index >= self.total_rois:
            raise IndexError(
                f"ROI index {index} out of range for total_rois={self.total_rois}"
            )
        return self.read_slice(index, index + 1)[0]

    def read_slice(self, start: int, end: int) -> np.ndarray:
        if start < 0 or end < start or end > self.total_rois:
            raise IndexError(
                f"Invalid ROI slice [{start}:{end}] for total_rois={self.total_rois}"
            )
        if self._roi_images is not None:
            assert self._roi_images is not None
            return _normalize_roi_batch(np.asarray(self._roi_images[start:end]))
        self._require_authoritative_live_pixel_contract()
        return self._read_live_indices(np.arange(start, end, dtype=np.int64))

    def read_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        roi_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if roi_indices.size == 0:
            roi_h, roi_w = self.roi_shape
            return np.zeros((0, roi_h, roi_w), dtype=np.uint8)
        if (
            roi_indices.min(initial=0) < 0
            or roi_indices.max(initial=0) >= self.total_rois
        ):
            raise IndexError("ROI indices out of range for crop run.")
        if self._roi_images is not None:
            assert self._roi_images is not None
            return _normalize_roi_batch(np.asarray(self._roi_images[roi_indices]))

        self._require_authoritative_live_pixel_contract()
        return self._read_live_indices(roi_indices)

    def _require_authoritative_live_pixel_contract(self) -> None:
        required = authoritative_crop_roi_pixel_contract(
            self.crop_group,
            run_id=self.crop_run_name,
        )
        if required is None:
            return
        observed = normalize_pixel_contract(self.roi_pixel_contract)
        if observed != required:
            raise ValueError(
                "Authoritative crop pixels require the "
                f"{required['name']!r} materialization contract; live reader "
                f"{None if observed is None else observed.get('name')!r} is not "
                "an allowed substitute. Build or open the bound flat ROI cache."
            )

    def _read_live_indices(self, roi_indices: np.ndarray) -> np.ndarray:
        if self.frame_source_kind == "acquisition_crop_video":
            return self._read_acquisition_crop_video_indices(roi_indices)
        if (
            self.frame_source_kind == "source_video_path"
            and self.frame_source_path is not None
            and self.roi_live_acceleration_effective == "gpu"
            and self.frame_shape is not None
        ):
            local_frames = self.frame_indices[roi_indices] - int(
                self.source_video_frame_offset
            )
            if np.any(local_frames < 0) or (
                self.source_video_frame_count is not None
                and np.any(local_frames >= int(self.source_video_frame_count))
            ):
                raise IndexError(
                    "Requested crop rows fall outside the bound source-video "
                    "acquisition-frame window."
                )
            try:
                return _read_external_video_live_gpu_batch(
                    video_path=Path(self.frame_source_path),
                    frame_indices=local_frames,
                    roi_coordinates_full=self.roi_coordinates_full[roi_indices],
                    roi_shape=self.roi_shape,
                    video_shape=self.frame_shape,
                    gpu_chunk_frames=self.roi_live_gpu_chunk_frames,
                )
            except Exception as exc:
                raise _gpu_decode_unavailable(
                    f"external-video live GPU read failed: {exc.__class__.__name__}: {exc}"
                ) from exc
        return self._read_live_indices_cpu(roi_indices)

    def _read_acquisition_crop_video_indices(
        self, roi_indices: np.ndarray
    ) -> np.ndarray:
        if (
            self._acquisition_crop_reader is None and not self._acquisition_crop_readers
        ) or self.crop_video_frame_indices is None:
            raise RuntimeError(
                "No acquisition crop-video reader available for crop run."
            )
        roi_h, roi_w = self.roi_shape
        batch = np.zeros((roi_indices.size, roi_h, roi_w), dtype=np.uint8)

        if self.source_pixel_kind_codes is not None:
            source_kind_codes = self.source_pixel_kind_codes[roi_indices]
            video_positions = np.flatnonzero(source_kind_codes == 0)
            supplemental_positions = np.flatnonzero(source_kind_codes != 0)
        else:
            video_positions = np.arange(roi_indices.size, dtype=np.int64)
            supplemental_positions = np.zeros(0, dtype=np.int64)

        frame_cache: dict[tuple[int, int], np.ndarray] = {}
        for batch_idx in video_positions:
            crop_row = int(roi_indices[int(batch_idx)])
            video_frame_idx = self.crop_video_frame_indices[crop_row]
            video_frame_idx_int = int(video_frame_idx)
            if video_frame_idx_int < 0:
                raise ValueError(
                    "Acquisition crop-video source row points at a negative "
                    f"source_crop_video_frame_indices value for crop row {int(roi_indices[int(batch_idx)])}."
                )
            if self.crop_video_member_indices is None:
                member_index = -1
                reader = self._acquisition_crop_reader
            else:
                member_index = int(self.crop_video_member_indices[crop_row])
                if member_index < 0:
                    raise ValueError(
                        "Acquisition crop-video source row points at a negative "
                        f"source_crop_video_member_indices value for crop row {crop_row}."
                    )
                reader = (self._acquisition_crop_readers or {}).get(member_index)
            if reader is None:
                raise RuntimeError(
                    f"No acquisition crop-video reader exists for member {member_index}."
                )
            cache_key = (member_index, video_frame_idx_int)
            frame = frame_cache.get(cache_key)
            if frame is None:
                frame = reader.read_frame(video_frame_idx_int)
                if frame.shape[:2] != (roi_h, roi_w):
                    raise ValueError(
                        "Acquisition crop-video frame shape "
                        f"{frame.shape[:2]} does not match crop run ROI shape {(roi_h, roi_w)} "
                        f"for frame {video_frame_idx_int}."
                    )
                frame_cache[cache_key] = frame
            batch[batch_idx] = frame

        if supplemental_positions.size:
            if (
                self._supplemental_flat_cache is None
                or self.supplemental_cache_row_indices is None
            ):
                raise RuntimeError(
                    "Hybrid acquisition crop run has supplemental rows but no flat cache."
                )
            cache_rows = self.supplemental_cache_row_indices[
                roi_indices[supplemental_positions]
            ]
            if np.any(cache_rows < 0):
                bad_crop_row = int(
                    roi_indices[
                        supplemental_positions[np.flatnonzero(cache_rows < 0)[0]]
                    ]
                )
                raise ValueError(
                    "Supplemental source row points at a negative supplemental_cache_row_indices "
                    f"value for crop row {bad_crop_row}."
                )
            supplemental = _normalize_roi_batch(
                np.asarray(self._supplemental_flat_cache[cache_rows])
            )
            if supplemental.shape[1:] != (roi_h, roi_w):
                raise ValueError(
                    "Supplemental flat-cache ROI shape "
                    f"{supplemental.shape[1:]} does not match crop run ROI shape {(roi_h, roi_w)}."
                )
            batch[supplemental_positions] = supplemental
        return batch

    def _read_live_indices_cpu(self, roi_indices: np.ndarray) -> np.ndarray:
        roi_h, roi_w = self.roi_shape
        batch = np.zeros((roi_indices.size, roi_h, roi_w), dtype=np.uint8)
        coords = self.roi_coordinates_full[roi_indices]
        frames = self.frame_indices[roi_indices]
        frame_cache: dict[int, np.ndarray] = {}

        for batch_idx, (frame_idx, top_left) in enumerate(zip(frames, coords)):
            frame_idx_int = int(frame_idx)
            frame = frame_cache.get(frame_idx_int)
            if frame is None:
                frame = self._read_frame(frame_idx_int)
                frame_cache[frame_idx_int] = frame
            batch[batch_idx] = _crop_from_top_left(frame, top_left, self.roi_shape)
        return batch

    def _activate_flat_bin_cache(
        self,
        *,
        manifest_path: Path,
        zarr_path: str | Path | None,
    ) -> None:
        archive_path = (
            Path(zarr_path).expanduser().resolve() if zarr_path is not None else None
        )
        expected_shape = (
            self.total_rois,
            int(self.roi_shape[0]),
            int(self.roi_shape[1]),
        )
        cache_arr = open_flat_roi_cache(
            manifest_path,
            expected_archive_path=archive_path,
            expected_crop_run=self.crop_run_name,
            expected_shape=expected_shape,
        )
        builder = cache_arr.manifest.get("builder")
        stored_contract = (
            normalize_pixel_contract(builder.get("pixel_contract"))
            if isinstance(builder, dict)
            else None
        )
        required_contract = authoritative_crop_roi_pixel_contract(
            self.crop_group,
            run_id=self.crop_run_name,
        )
        if required_contract is not None and stored_contract != required_contract:
            cache_arr.close()
            raise ValueError(
                "Flat ROI cache pixel contract does not match the authoritative "
                f"crop source: expected {required_contract['name']!r}, observed "
                f"{None if stored_contract is None else stored_contract.get('name')!r}."
            )
        self._roi_images = cache_arr
        self.roi_cache_used = True
        self.roi_cache_created = False
        self.roi_cache_key = str(cache_arr.manifest.get("cache_key") or "") or None
        self.roi_cache_path = str(cache_arr.manifest_path)
        staging = cache_arr.manifest.get("staging")
        if isinstance(staging, dict):
            requested_manifest = staging.get("requested_manifest_path")
            self.roi_cache_canonical_path = (
                str(requested_manifest)
                if requested_manifest
                else str(cache_arr.manifest_path)
            )
        else:
            self.roi_cache_canonical_path = str(cache_arr.manifest_path)
        self.roi_cache_backend = "flat_bin_v1"
        self.roi_read_mode = "flat_bin_roi_cache"
        if isinstance(builder, dict):
            contract = stored_contract
            if contract is not None:
                self.roi_pixel_contract = contract
                self.roi_image_representation = _image_representation_from_contract(
                    contract
                )

    def _should_use_roi_cache(self) -> bool:
        if self.storage_mode != "geometry_only":
            return False
        if self.frame_source_kind == "acquisition_crop_video":
            return False
        if self.roi_cache_policy == "never":
            return False
        if self.roi_cache_policy == "always":
            return True
        if self.frame_source_kind == "source_video_path":
            return True
        if self.frame_shape is None:
            return False
        frame_pixels = int(self.frame_shape[0]) * int(self.frame_shape[1])
        return frame_pixels >= _ROI_CACHE_AUTO_MIN_SOURCE_PIXELS

    def _activate_temporary_cache(
        self,
        *,
        zarr_path: str | Path | None,
        roi_cache_dir: str | Path | None,
        console: Any | None = None,
    ) -> None:
        archive_path: Path | None = None
        if zarr_path is not None:
            archive_path = Path(zarr_path).expanduser().resolve()

        if archive_path is None:
            store_path = getattr(getattr(self.root, "store", None), "path", None)
            if store_path:
                archive_path = Path(str(store_path)).expanduser().resolve()

        if archive_path is None:
            if self.roi_cache_policy == "always":
                raise ValueError(
                    "Temporary ROI cache requires a stable zarr_path or store path to compute cache identity."
                )
            return

        cache_key = self._build_roi_cache_key(archive_path)
        cache_root = _resolve_roi_cache_root(roi_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_name = (
            f"{_cache_component(archive_path.stem, default='archive')}"
            f"__{_cache_component(self.crop_run_name, default='crop')}"
            f"__{cache_key[:12]}.zarr"
        )
        cache_path = cache_root / cache_name
        cache_group = zarr.open_group(str(cache_path), mode="a")

        expected_shape = (
            self.total_rois,
            int(self.roi_shape[0]),
            int(self.roi_shape[1]),
        )
        cache_complete = bool(cache_group.attrs.get("cache_complete"))
        cache_key_before = _normalize_run_name(cache_group.attrs.get("cache_key"))
        cache_arr = cache_group.get("roi_images")
        reuse_existing = (
            cache_complete
            and cache_key_before == cache_key
            and cache_arr is not None
            and tuple(getattr(cache_arr, "shape", ())) == expected_shape
        )

        if not reuse_existing:
            cache_group.attrs.update(
                {
                    "cache_complete": False,
                    "cache_key": cache_key,
                    "archive_path": str(archive_path),
                    "crop_run_name": self.crop_run_name,
                    "source_crop_storage_mode": self.storage_mode,
                    "frame_source_kind": self.frame_source_kind,
                    "frame_source_path": self.frame_source_path,
                    "roi_shape": [int(self.roi_shape[0]), int(self.roi_shape[1])],
                    "total_rois": int(self.total_rois),
                    "crop_signature": self.crop_group.attrs.get("crop_signature"),
                    "crop_run_reference": build_crop_run_reference(
                        self.crop_group,
                        run_id=self.crop_run_name,
                        allow_unversioned_legacy=True,
                    ),
                    "roi_image_representation": self.roi_image_representation,
                    "roi_pixel_contract": self.roi_pixel_contract,
                }
            )
            if console is not None and hasattr(console, "print"):
                console.print(
                    f"[cyan]Building temporary ROI cache[/cyan] [dim](crop_run={self.crop_run_name})[/dim]"
                )
                console.print(f"[dim]{cache_path}[/dim]")
            if self.frame_source_kind == "source_video_path" and self.frame_source_path:
                from fisheye.tracking.crop import (
                    materialize_external_roi_cache_for_crop_run,
                )

                cache_result = materialize_external_roi_cache_for_crop_run(
                    cache_path=cache_path,
                    source_zarr_path=archive_path,
                    crop_run_name=self.crop_run_name,
                    console=console,
                    write_backend="kvikio",
                    roi_storage="uncompressed",
                    use_sharding=False,
                    roi_chunk_size=DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
                    roi_shard_size=None,
                    gpu_chunk_frames=DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
                    require_kvikio=False,
                    prefer_gpu=True,
                )
                cache_group = zarr.open_group(str(cache_path), mode="a")
                cache_arr = cache_group.get("roi_images")
                cache_group.attrs.update(
                    {
                        "cache_write_backend_requested": cache_result.get(
                            "write_backend_requested"
                        ),
                        "cache_write_backend_effective": cache_result.get(
                            "write_backend_effective"
                        ),
                        "cache_acceleration": cache_result.get("acceleration"),
                        "cache_fallback_reason": cache_result.get("fallback_reason"),
                        "cache_decode_seconds": cache_result.get("decode_seconds"),
                        "cache_compute_seconds": cache_result.get("compute_seconds"),
                        "cache_write_seconds": cache_result.get("write_seconds"),
                        "cache_duration_seconds": cache_result.get("duration_seconds"),
                        "cache_roi_chunk_len": cache_result.get("roi_chunk_len"),
                        "cache_roi_shard_len": cache_result.get("roi_shard_len"),
                        "cache_roi_storage": cache_result.get("roi_storage"),
                        "cache_roi_use_sharding": cache_result.get("roi_use_sharding"),
                        "cache_layout_profile": cache_result.get(
                            "roi_layout_profile", SCRATCH_ROI_CACHE_LAYOUT_PROFILE
                        ),
                        "cache_gpu_chunk_frames": cache_result.get("gpu_chunk_frames"),
                        "roi_image_representation": cache_result.get(
                            "roi_image_representation"
                        ),
                        "roi_pixel_contract": cache_result.get("roi_pixel_contract"),
                    }
                )
            else:
                cache_layout = build_scratch_roi_cache_layout(
                    total_rois=self.total_rois
                )
                cache_arr = cache_group.create_array(
                    "roi_images",
                    **build_crop_roi_create_kwargs(
                        total_rois=self.total_rois,
                        roi_sz=self.roi_shape,
                        layout=cache_layout,
                        overwrite=True,
                    ),
                )
                if console is not None and hasattr(console, "print"):
                    console.print(
                        "[dim]Temporary ROI cache materialization: "
                        "backend=standard_zarr, acceleration=cpu, source=raw_video/images_full[/dim]"
                    )
                total_batches = max(
                    1,
                    (self.total_rois + _ROI_CACHE_BUILD_BATCH - 1)
                    // _ROI_CACHE_BUILD_BATCH,
                )
                progress_every = max(1, total_batches // 20)
                for batch_idx, start in enumerate(
                    range(0, self.total_rois, _ROI_CACHE_BUILD_BATCH)
                ):
                    end = min(start + _ROI_CACHE_BUILD_BATCH, self.total_rois)
                    cache_arr[start:end] = self._read_live_indices(
                        np.arange(start, end, dtype=np.int64)
                    )
                    if (
                        console is not None
                        and hasattr(console, "print")
                        and (
                            end == self.total_rois
                            or (batch_idx + 1) % progress_every == 0
                        )
                    ):
                        pct = (
                            (end / self.total_rois) * 100
                            if self.total_rois > 0
                            else 100.0
                        )
                        console.print(
                            f"[dim]  Cache progress: {end:,}/{self.total_rois:,} ROIs ({pct:.1f}%)[/dim]"
                        )
                cache_group.attrs["cache_write_backend_requested"] = "standard"
                cache_group.attrs["cache_write_backend_effective"] = "standard_zarr"
                cache_group.attrs["cache_acceleration"] = "cpu"
                cache_group.attrs["cache_roi_chunk_len"] = int(
                    cache_layout.roi_chunk_len
                )
                cache_group.attrs["cache_roi_shard_len"] = (
                    int(cache_layout.roi_shard_len)
                    if cache_layout.roi_shard_len is not None
                    else int(cache_layout.roi_chunk_len)
                )
                cache_group.attrs["cache_roi_storage"] = cache_layout.roi_storage
                cache_group.attrs["cache_roi_use_sharding"] = bool(
                    cache_layout.roi_use_sharding
                )
                cache_group.attrs["cache_layout_profile"] = (
                    SCRATCH_ROI_CACHE_LAYOUT_PROFILE
                )
                cache_group.attrs.update(crop_roi_layout_attrs(cache_layout))
            cache_group.attrs["cache_complete"] = True
            if console is not None and hasattr(console, "print"):
                console.print(
                    f"[green]Temporary ROI cache ready[/green] "
                    f"[dim](crop_run={self.crop_run_name}, {_cache_runtime_summary(cache_group)})[/dim]"
                )
                console.print(f"[dim]{cache_path}[/dim]")
        elif console is not None and hasattr(console, "print"):
            console.print(
                f"[green]Reusing temporary ROI cache[/green] "
                f"[dim](crop_run={self.crop_run_name}, {_cache_runtime_summary(cache_group)})[/dim]"
            )
            console.print(f"[dim]{cache_path}[/dim]")

        self._roi_images = cache_arr
        self.roi_cache_used = True
        self.roi_cache_created = not reuse_existing
        self.roi_cache_key = cache_key
        self.roi_cache_path = str(cache_path)
        self.roi_cache_backend = "zarr"
        self.roi_read_mode = "temporary_cache"
        contract = normalize_pixel_contract(cache_group.attrs.get("roi_pixel_contract"))
        if contract is not None:
            self.roi_pixel_contract = contract
            self.roi_image_representation = _image_representation_from_contract(
                contract
            )

    def _build_roi_cache_key(self, archive_path: Path) -> str:
        payload = {
            "schema": "palette_roi_cache_v1",
            "archive_path": str(archive_path),
            "crop_run_name": self.crop_run_name,
            "crop_run_reference": build_crop_run_reference(
                self.crop_group,
                run_id=self.crop_run_name,
                allow_unversioned_legacy=True,
            ),
            "frame_source_kind": self.frame_source_kind,
            "frame_source_identity": self._build_frame_source_identity(),
            "roi_shape": [int(self.roi_shape[0]), int(self.roi_shape[1])],
            "total_rois": int(self.total_rois),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def _build_frame_source_identity(self) -> dict[str, object]:
        identity: dict[str, object] = {
            "frame_source_path": self.frame_source_path,
            "frame_shape": (
                list(self.frame_shape) if self.frame_shape is not None else None
            ),
        }
        if self.frame_source_path_override_used:
            windowed = self.source_video_frame_count is not None
            identity.update(
                {
                    "frame_source_declared_path": self.frame_source_declared_path,
                    "frame_source_path_override_used": True,
                    "override_semantics": (
                        "acquisition_frame_window_relocation_v1"
                        if windowed
                        else "byte-identical_relocation_for_cache_materialization"
                    ),
                    "source_video_frame_offset": int(self.source_video_frame_offset),
                    "source_video_frame_count": self.source_video_frame_count,
                }
            )

        if (
            self.frame_source_kind in {"source_video_path", "acquisition_crop_video"}
            and self.frame_source_path
        ):
            source_path = Path(self.frame_source_path)
            try:
                stat = source_path.stat()
            except OSError:
                return identity
            identity.update(
                {
                    "source_path_exists": True,
                    "source_path_size": int(stat.st_size),
                    "source_path_mtime_ns": int(stat.st_mtime_ns),
                }
            )
            return identity

        if self._images_full is not None:
            identity.update(
                {
                    "images_full_shape": list(getattr(self._images_full, "shape", ())),
                    "images_full_dtype": str(getattr(self._images_full, "dtype", "")),
                    "images_full_path": getattr(self._images_full, "path", None),
                }
            )
        return identity

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        if self._images_full is not None:
            if frame_idx < 0 or frame_idx >= int(self._images_full.shape[0]):
                raise IndexError(
                    f"Frame index {frame_idx} exceeds raw_video/images_full length ({self._images_full.shape[0]})."
                )
            return _to_grayscale_uint8(np.asarray(self._images_full[frame_idx]))
        if self._external_reader is not None:
            local_frame = int(frame_idx) - int(self.source_video_frame_offset)
            if local_frame < 0:
                raise IndexError(
                    f"Acquisition frame {frame_idx} precedes source-video window "
                    f"offset {self.source_video_frame_offset}."
                )
            if self.source_video_frame_count is not None and local_frame >= int(
                self.source_video_frame_count
            ):
                raise IndexError(
                    f"Acquisition frame {frame_idx} exceeds source-video window "
                    f"[{self.source_video_frame_offset}, "
                    f"{self.source_video_frame_offset + self.source_video_frame_count})."
                )
            return self._external_reader.read_frame(local_frame)
        if self._acquisition_crop_reader is not None:
            return self._acquisition_crop_reader.read_frame(frame_idx)
        raise RuntimeError("No frame source available for geometry-only crop read.")

    def close(self) -> None:
        if self._roi_images is not None and hasattr(self._roi_images, "close"):
            self._roi_images.close()
        if self._supplemental_flat_cache is not None and hasattr(
            self._supplemental_flat_cache,
            "close",
        ):
            self._supplemental_flat_cache.close()
        if self._external_reader is not None:
            self._external_reader.close()
        if self._acquisition_crop_reader is not None:
            self._acquisition_crop_reader.close()
        if self._acquisition_crop_readers is not None:
            for reader in self._acquisition_crop_readers.values():
                reader.close()
