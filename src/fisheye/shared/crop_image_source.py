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

from fisheye.shared.crop_roi_layout import (
    DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
    DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
    SCRATCH_ROI_CACHE_LAYOUT_PROFILE,
    build_crop_roi_create_kwargs,
    build_scratch_roi_cache_layout,
    crop_roi_layout_attrs,
)
from fisheye.shared.flat_roi_cache import crop_run_name_from_manifest, open_flat_roi_cache
from fisheye.shared.roi_pixel_contract import (
    ROI_IMAGE_REPRESENTATION,
    crop_image_source_live_pixel_contract,
    crop_run_pixel_contract,
    normalize_pixel_contract,
)
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_run_completion import is_run_complete

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

_ROI_CACHE_POLICIES = {"never", "auto", "always"}
_ROI_CACHE_AUTO_MIN_SOURCE_PIXELS = 2048 * 2048
_ROI_CACHE_BUILD_BATCH = 64
_ROI_LIVE_ACCELERATION_CHOICES = {"auto", "cpu", "gpu"}
_ROI_LIVE_GPU_CHUNK_FRAMES_DEFAULT = 32


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
        if candidate and candidate in crop_parent and is_run_complete(crop_parent[candidate], legacy_default=True):
            return crop_parent, crop_parent[candidate], candidate

    raise ValueError("No crop run found; cannot resolve latest_any/latest/latest_materialized")


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
            raise ValueError(
                f"Crop run '{run_name}' is not materialized. Traditional pipelines require "
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
        if candidate and candidate in crop_parent and is_run_complete(crop_parent[candidate], legacy_default=True):
            run_group = crop_parent[candidate]
            if _resolve_storage_mode(run_group) == "materialized" and "roi_images" in run_group:
                return crop_parent, run_group, candidate

    latest_any = _normalize_run_name(crop_parent.attrs.get("latest_any"))
    if latest_any and latest_any in crop_parent and is_run_complete(crop_parent[latest_any], legacy_default=True):
        run_group = crop_parent[latest_any]
        if _resolve_storage_mode(run_group) != "materialized":
            raise ValueError(
                f"Latest available crop run '{latest_any}' is geometry-only. Traditional pipelines require "
                "a materialized crop run with roi_images."
            )

    raise ValueError(
        "No materialized crop run found under crop_runs. Traditional pipelines require crop_runs/<run>/roi_images."
    )


def _normalize_storage_mode(value: object) -> str | None:
    text = _normalize_run_name(value)
    if text in {"materialized", "geometry_only"}:
        return text
    return None


def _resolve_storage_mode(crop_group: zarr.Group) -> str:
    explicit = _normalize_storage_mode(crop_group.attrs.get("crop_storage_mode"))
    if explicit is not None:
        return explicit
    if "roi_images" in crop_group:
        return "materialized"
    return "geometry_only"


def _resolve_roi_shape(crop_group: zarr.Group) -> tuple[int, int]:
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    if "roi_images" in crop_group:
        roi_images = crop_group["roi_images"]
        if len(roi_images.shape) < 3:
            raise ValueError("crop_runs/<run>/roi_images must be at least 3D")
        return int(roi_images.shape[1]), int(roi_images.shape[2])
    raise ValueError("Unable to determine ROI size for crop run (need roi_size or roi_images).")


def _normalize_roi_cache_policy(value: object) -> str:
    text = _normalize_run_name(value) or "never"
    if text not in _ROI_CACHE_POLICIES:
        choices = ", ".join(sorted(_ROI_CACHE_POLICIES))
        raise ValueError(f"Invalid roi_cache_policy '{text}'. Expected one of: {choices}")
    return text


def _normalize_roi_live_acceleration(value: object) -> str:
    text = _normalize_run_name(value) or "auto"
    if text not in _ROI_LIVE_ACCELERATION_CHOICES:
        choices = ", ".join(sorted(_ROI_LIVE_ACCELERATION_CHOICES))
        raise ValueError(f"Invalid roi_live_acceleration '{text}'. Expected one of: {choices}")
    return text


def _resolve_frame_shape(
    root: zarr.Group,
    crop_group: zarr.Group,
    images_full: object | None,
) -> tuple[int, int] | None:
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
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text.strip())
    return cleaned or default


def _cache_runtime_summary(cache_group: zarr.Group) -> str:
    backend = normalize_attr(cache_group.attrs.get("cache_write_backend_effective")) or "standard_zarr"
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
        gray = (
            0.299 * arr[..., 0].astype(np.float32)
            + 0.587 * arr[..., 1].astype(np.float32)
            + 0.114 * arr[..., 2].astype(np.float32)
        )
        return gray.astype(np.uint8)
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
    stored = (
        normalize_pixel_contract(crop_group.attrs.get("roi_pixel_contract"))
        or normalize_pixel_contract(crop_group.attrs.get("crop_pixel_contract"))
    )
    if stored is not None and crop_storage_mode == "materialized":
        return stored
    if crop_storage_mode == "materialized":
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

    if vy2 - vy1 == roi_h and vx2 - vx1 == roi_w and 0 <= y1 and 0 <= x1 and y2 <= height and x2 <= width:
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

    if tracking_crop.VideoReader is None or tracking_crop.gpu is None or tracking_crop.decord is None:
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
        video_reader = tracking_crop.VideoReader(str(video_path), ctx=tracking_crop.gpu(0))
        for chunk_idx, start in enumerate(range(0, len(unique_frames), chunk_len)):
            chunk_frames = unique_frames[start : start + chunk_len]
            if not chunk_frames:
                continue
            frames_gpu = video_reader.get_batch(chunk_frames)
            _, roi_ids, crops_cpu, _coords_cpu, _chunk_time = tracking_crop._process_chunk_gpu_from_top_left(
                chunk_idx,
                chunk_frames,
                frames_gpu,
                frame_to_roi,
                roi_coordinates_np,
                roi_shape,
                video_shape,
                return_device=False,
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
        raise RuntimeError(f"Unable to open source video for live ROI reads: {self.video_path}")

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
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return _to_grayscale_uint8(frame)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._reader = None


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
    frame_shape: tuple[int, int] | None = None
    roi_read_mode: str = "materialized_crop_run"
    roi_cache_policy: str = "never"
    roi_cache_used: bool = False
    roi_cache_created: bool = False
    roi_cache_key: str | None = None
    roi_cache_path: str | None = None
    roi_cache_backend: str | None = None
    roi_image_representation: str | None = ROI_IMAGE_REPRESENTATION
    roi_pixel_contract: dict[str, Any] | None = None
    roi_live_acceleration_requested: str | None = None
    roi_live_acceleration_effective: str | None = None
    roi_live_acceleration_fallback_reason: str | None = None
    roi_live_gpu_chunk_frames: int = _ROI_LIVE_GPU_CHUNK_FRAMES_DEFAULT
    _roi_images: object | None = None
    _images_full: object | None = None
    _external_reader: _ExternalFrameReader | None = None

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
        console: Any | None = None,
    ) -> "CropImageSource":
        normalized_cache_policy = _normalize_roi_cache_policy(roi_cache_policy)
        normalized_live_acceleration = _normalize_roi_live_acceleration(roi_live_acceleration)
        live_gpu_chunk_frames = max(1, int(roi_live_gpu_chunk_frames))
        manifest_path = Path(roi_cache_manifest).expanduser() if roi_cache_manifest is not None else None
        if crop_run is None and manifest_path is not None:
            crop_run = crop_run_name_from_manifest(manifest_path)
        _crop_parent, crop_group, crop_run_name = resolve_crop_run(
            root,
            crop_run=crop_run,
            zarr_path=zarr_path,
        )
        storage_mode = _resolve_storage_mode(crop_group)
        roi_shape = _resolve_roi_shape(crop_group)

        if "roi_coordinates_full" not in crop_group:
            raise ValueError("Crop run missing 'roi_coordinates_full'.")
        roi_coordinates_full = np.asarray(crop_group["roi_coordinates_full"][:], dtype=np.int32)
        total_rois = int(roi_coordinates_full.shape[0])

        frame_indices_arr = crop_group.get("frame_indices")
        if frame_indices_arr is not None:
            frame_indices = np.asarray(frame_indices_arr[:], dtype=np.int64)
        elif storage_mode == "materialized":
            frame_indices = np.zeros(total_rois, dtype=np.int64)
        else:
            raise ValueError("Geometry-only crop run is missing 'frame_indices'.")

        if frame_indices.shape[0] != total_rois:
            raise ValueError(
                f"frame_indices length {frame_indices.shape[0]} does not match roi_coordinates_full length {total_rois}"
            )

        roi_images = crop_group.get("roi_images")
        if storage_mode == "materialized":
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
        else:
            raw_video = root.get("raw_video")
            images_full = raw_video.get("images_full") if raw_video is not None else None
            frame_shape = _resolve_frame_shape(root, crop_group, images_full)
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
                source_video_path = (
                    crop_group.attrs.get("source_video_path")
                    or crop_group.attrs.get("video_source_path")
                    or root.attrs.get("source_video_path")
                )
                if not source_video_path:
                    raise ValueError(
                        "Geometry-only crop run requires raw_video/images_full or source_video_path provenance."
                    )
                frame_source_kind = "source_video_path"
                frame_source_path = str(source_video_path)
                external_reader = _ExternalFrameReader(Path(frame_source_path))
                if normalized_live_acceleration == "cpu":
                    live_acceleration_effective = "cpu"
                    live_acceleration_fallback_reason = None
                else:
                    gpu_available, gpu_reason = _check_external_video_live_gpu_available()
                    if normalized_live_acceleration == "gpu":
                        if not gpu_available:
                            raise ValueError(
                                "roi_live_acceleration='gpu' requested, but GPU live ROI reads are unavailable: "
                                f"{gpu_reason}"
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
                        live_acceleration_effective = "cpu"
                        if frame_shape is None:
                            live_acceleration_fallback_reason = "unknown_frame_shape"
                        else:
                            live_acceleration_fallback_reason = gpu_reason
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
            frame_shape=frame_shape,
            roi_read_mode=roi_read_mode,
            roi_cache_policy=normalized_cache_policy,
            roi_image_representation=_image_representation_from_contract(roi_pixel_contract),
            roi_pixel_contract=roi_pixel_contract,
            roi_live_acceleration_requested=(
                normalized_live_acceleration if storage_mode == "geometry_only" else None
            ),
            roi_live_acceleration_effective=(
                live_acceleration_effective if storage_mode == "geometry_only" else None
            ),
            roi_live_acceleration_fallback_reason=(
                live_acceleration_fallback_reason if storage_mode == "geometry_only" else None
            ),
            roi_live_gpu_chunk_frames=live_gpu_chunk_frames,
            _roi_images=roi_images,
            _images_full=images_full if storage_mode == "geometry_only" else None,
            _external_reader=external_reader if storage_mode == "geometry_only" else None,
        )
        if manifest_path is not None:
            source._activate_flat_bin_cache(
                manifest_path=manifest_path,
                zarr_path=zarr_path,
            )
        elif source.storage_mode == "geometry_only" and source._should_use_roi_cache():
            source._activate_temporary_cache(
                zarr_path=zarr_path,
                roi_cache_dir=roi_cache_dir,
                console=console,
            )
        return source

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
            raise IndexError(f"ROI index {index} out of range for total_rois={self.total_rois}")
        return self.read_slice(index, index + 1)[0]

    def read_slice(self, start: int, end: int) -> np.ndarray:
        if start < 0 or end < start or end > self.total_rois:
            raise IndexError(f"Invalid ROI slice [{start}:{end}] for total_rois={self.total_rois}")
        if self._roi_images is not None:
            assert self._roi_images is not None
            return _normalize_roi_batch(np.asarray(self._roi_images[start:end]))
        return self._read_live_indices(np.arange(start, end, dtype=np.int64))

    def read_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        roi_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if roi_indices.size == 0:
            roi_h, roi_w = self.roi_shape
            return np.zeros((0, roi_h, roi_w), dtype=np.uint8)
        if roi_indices.min(initial=0) < 0 or roi_indices.max(initial=0) >= self.total_rois:
            raise IndexError("ROI indices out of range for crop run.")
        if self._roi_images is not None:
            assert self._roi_images is not None
            return _normalize_roi_batch(np.asarray(self._roi_images[roi_indices]))

        return self._read_live_indices(roi_indices)

    def _read_live_indices(self, roi_indices: np.ndarray) -> np.ndarray:
        if (
            self.frame_source_kind == "source_video_path"
            and self.frame_source_path is not None
            and self.roi_live_acceleration_effective == "gpu"
            and self.frame_shape is not None
        ):
            try:
                return _read_external_video_live_gpu_batch(
                    video_path=Path(self.frame_source_path),
                    frame_indices=self.frame_indices[roi_indices],
                    roi_coordinates_full=self.roi_coordinates_full[roi_indices],
                    roi_shape=self.roi_shape,
                    video_shape=self.frame_shape,
                    gpu_chunk_frames=self.roi_live_gpu_chunk_frames,
                )
            except Exception as exc:
                if self.roi_live_acceleration_requested == "gpu":
                    raise
                self.roi_live_acceleration_effective = "cpu"
                self.roi_live_acceleration_fallback_reason = (
                    f"runtime_gpu_fallback:{exc.__class__.__name__}: {exc}"
                )
        return self._read_live_indices_cpu(roi_indices)

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
        archive_path = Path(zarr_path).expanduser().resolve() if zarr_path is not None else None
        expected_shape = (self.total_rois, int(self.roi_shape[0]), int(self.roi_shape[1]))
        cache_arr = open_flat_roi_cache(
            manifest_path,
            expected_archive_path=archive_path,
            expected_crop_run=self.crop_run_name,
            expected_shape=expected_shape,
        )
        self._roi_images = cache_arr
        self.roi_cache_used = True
        self.roi_cache_created = False
        self.roi_cache_key = str(cache_arr.manifest.get("cache_key") or "") or None
        self.roi_cache_path = str(cache_arr.manifest_path)
        self.roi_cache_backend = "flat_bin_v1"
        self.roi_read_mode = "flat_bin_roi_cache"
        builder = cache_arr.manifest.get("builder")
        if isinstance(builder, dict):
            contract = normalize_pixel_contract(builder.get("pixel_contract"))
            if contract is not None:
                self.roi_pixel_contract = contract
                self.roi_image_representation = _image_representation_from_contract(contract)

    def _should_use_roi_cache(self) -> bool:
        if self.storage_mode != "geometry_only":
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

        expected_shape = (self.total_rois, int(self.roi_shape[0]), int(self.roi_shape[1]))
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
                from fisheye.tracking.crop import materialize_external_roi_cache_for_crop_run

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
                        "cache_write_backend_requested": cache_result.get("write_backend_requested"),
                        "cache_write_backend_effective": cache_result.get("write_backend_effective"),
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
                        "cache_layout_profile": cache_result.get("roi_layout_profile", SCRATCH_ROI_CACHE_LAYOUT_PROFILE),
                        "cache_gpu_chunk_frames": cache_result.get("gpu_chunk_frames"),
                        "roi_image_representation": cache_result.get("roi_image_representation"),
                        "roi_pixel_contract": cache_result.get("roi_pixel_contract"),
                    }
                )
            else:
                cache_layout = build_scratch_roi_cache_layout(total_rois=self.total_rois)
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
                total_batches = max(1, (self.total_rois + _ROI_CACHE_BUILD_BATCH - 1) // _ROI_CACHE_BUILD_BATCH)
                progress_every = max(1, total_batches // 20)
                for batch_idx, start in enumerate(range(0, self.total_rois, _ROI_CACHE_BUILD_BATCH)):
                    end = min(start + _ROI_CACHE_BUILD_BATCH, self.total_rois)
                    cache_arr[start:end] = self._read_live_indices(np.arange(start, end, dtype=np.int64))
                    if console is not None and hasattr(console, "print") and (
                        end == self.total_rois or (batch_idx + 1) % progress_every == 0
                    ):
                        pct = (end / self.total_rois) * 100 if self.total_rois > 0 else 100.0
                        console.print(
                            f"[dim]  Cache progress: {end:,}/{self.total_rois:,} ROIs ({pct:.1f}%)[/dim]"
                        )
                cache_group.attrs["cache_write_backend_requested"] = "standard"
                cache_group.attrs["cache_write_backend_effective"] = "standard_zarr"
                cache_group.attrs["cache_acceleration"] = "cpu"
                cache_group.attrs["cache_roi_chunk_len"] = int(cache_layout.roi_chunk_len)
                cache_group.attrs["cache_roi_shard_len"] = (
                    int(cache_layout.roi_shard_len) if cache_layout.roi_shard_len is not None else int(cache_layout.roi_chunk_len)
                )
                cache_group.attrs["cache_roi_storage"] = cache_layout.roi_storage
                cache_group.attrs["cache_roi_use_sharding"] = bool(cache_layout.roi_use_sharding)
                cache_group.attrs["cache_layout_profile"] = SCRATCH_ROI_CACHE_LAYOUT_PROFILE
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
            self.roi_image_representation = _image_representation_from_contract(contract)

    def _build_roi_cache_key(self, archive_path: Path) -> str:
        payload = {
            "schema": "palette_roi_cache_v1",
            "archive_path": str(archive_path),
            "crop_run_name": self.crop_run_name,
            "crop_signature": self.crop_group.attrs.get("crop_signature"),
            "frame_source_kind": self.frame_source_kind,
            "frame_source_identity": self._build_frame_source_identity(),
            "roi_shape": [int(self.roi_shape[0]), int(self.roi_shape[1])],
            "total_rois": int(self.total_rois),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _build_frame_source_identity(self) -> dict[str, object]:
        identity: dict[str, object] = {
            "frame_source_path": self.frame_source_path,
            "frame_shape": list(self.frame_shape) if self.frame_shape is not None else None,
        }

        if self.frame_source_kind == "source_video_path" and self.frame_source_path:
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
            return self._external_reader.read_frame(frame_idx)
        raise RuntimeError("No frame source available for geometry-only crop read.")

    def close(self) -> None:
        if self._roi_images is not None and hasattr(self._roi_images, "close"):
            self._roi_images.close()
        if self._external_reader is not None:
            self._external_reader.close()
