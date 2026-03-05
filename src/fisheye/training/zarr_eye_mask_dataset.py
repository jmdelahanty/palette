"""Zarr-backed dataset for eye-mask segmentation training."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import zarr
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from torch.utils.data import Dataset

from .config import (
    EyeMaskTrainingConfig,
    EyeMaskDatasetConfig,
    DatasetSplit,
    EyeMaskCacheConfig,
    CacheBackend,
)


@dataclass
class _EyeMaskEntry:
    dataset_name: str
    zarr_path: str
    crop_run: str
    mask_run: Optional[str]
    roi_images: Optional[zarr.Array]
    masks: Optional[zarr.Array]
    roi_coordinates_full: Optional[np.ndarray]
    background_full: Optional[np.ndarray]
    roi_index: int
    roi_shape: Tuple[int, int]
    has_positive: bool
    entry_type: str  # "positive" or "background"


@dataclass
class EyeMaskDatasetStats:
    dataset_name: str
    zarr_path: str
    crop_run: str
    mask_run: str
    total_rois: int
    positive_rois: int
    negative_rois: int
    background_negatives: int


@dataclass
class DatasetCacheInfo:
    dataset_name: str
    zarr_path: str
    crop_run: str
    mask_run: str
    include_background: bool
    background_run: Optional[str]
    background_array: Optional[str]

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "dataset_name": self.dataset_name,
            "zarr_path": self.zarr_path,
            "crop_run": self.crop_run,
            "mask_run": self.mask_run,
            "include_background": self.include_background,
            "background_run": self.background_run,
            "background_array": self.background_array,
        }


@dataclass
class YOLOTargetStore:
    roi_array: zarr.Array
    mask_probs_array: Optional[zarr.Array]
    masks_array: Optional[zarr.Array]
    use_probs: bool
    meta: Dict[str, object]


_CACHE_PROCESS_IMAGES: Optional[np.memmap] = None
_CACHE_PROCESS_MASKS: Optional[np.memmap] = None
_CACHE_TARGET_SIZE: int = 0
_CACHE_MASK_CHANNELS: int = 1
_CACHE_CONTEXTS: Dict[str, Dict[str, zarr.Array]] = {}


def load_yolo_targets(
    zarr_path: Union[str, Path],
    eye_masks_run: Optional[str],
    label_mode: str,
    eye_masks_method: Optional[str] = None,
    mask_preference: str = "auto",
    crop_run_override: Optional[str] = None,
) -> YOLOTargetStore:
    """Resolve ROI and mask arrays for U-Net training without materializing them."""

    zarr_path_str = str(zarr_path)
    root = zarr.open(zarr_path_str, mode="r")

    if "eye_masks_runs" not in root:
        raise ValueError(f"'eye_masks_runs' group not found in Zarr store: {zarr_path}")
    eye_parent = root["eye_masks_runs"]

    run_name: Optional[str] = eye_masks_run or eye_parent.attrs.get("latest")
    if (not run_name or run_name not in eye_parent) and eye_masks_method:
        candidates: List[str] = []
        for name in eye_parent.group_keys():
            grp = eye_parent[name]
            if getattr(grp, "attrs", None) and grp.attrs.get("method") == eye_masks_method:
                candidates.append(name)
        if candidates:
            candidates.sort()
            run_name = candidates[-1]
    if run_name is None:
        run_name = eye_parent.attrs.get("latest")
    if not run_name:
        raise ValueError("YOLO eye mask run not found (no run specified and no 'latest' attr)")
    if run_name not in eye_parent:
        raise ValueError(f"YOLO eye mask run '{run_name}' not present in Zarr store")
    run_grp = eye_parent[run_name]

    preference = str(mask_preference or "auto").strip().lower()
    if preference == "refined_probs":
        source_priority = ["mask_probs_roi_refined"]
    elif preference == "raw_probs":
        source_priority = ["mask_probs_roi", "mask_probs_roi_refined", "masks_roi"]
    elif preference == "binary":
        source_priority = ["masks_roi"]
    else:
        source_priority = ["mask_probs_roi_refined", "mask_probs_roi", "masks_roi"]

    mask_probs_arr: Optional[zarr.Array] = None
    masks_arr: Optional[zarr.Array] = None
    target_source: Optional[str] = None
    for candidate in source_priority:
        if candidate in run_grp:
            if candidate in {"mask_probs_roi", "mask_probs_roi_refined"}:
                mask_probs_arr = run_grp[candidate]
                masks_arr = None
                target_source = candidate
                break
            if candidate == "masks_roi":
                masks_arr = run_grp[candidate]
                mask_probs_arr = None
                target_source = candidate
                break

    if target_source is None:
        raise ValueError(
            f"YOLO eye mask run '{run_name}' does not provide requested mask data (mask_preference='{preference}')"
        )

    valid_modes = {"union", "lr"}
    if label_mode not in valid_modes:
        raise ValueError(f"label_mode='{label_mode}' not in {valid_modes}")

    use_probs = mask_probs_arr is not None

    crop_run = crop_run_override or run_grp.attrs.get("source_crop_run")
    if not crop_run:
        crop_parent = root.get("crop_runs", None)
        if crop_parent is not None:
            crop_run = crop_parent.attrs.get("latest")
    if not crop_run:
        raise ValueError(
            f"Missing 'source_crop_run' attribute for eye mask run '{run_name}' and no latest crop run recorded"
        )

    roi_path = f"crop_runs/{crop_run}/roi_images"
    if roi_path not in root:
        raise ValueError(f"ROI images array '{roi_path}' not found in Zarr store")

    roi_arr = root[roi_path]
    if roi_arr.ndim not in (3, 4):
        raise ValueError(f"Unexpected ROI images shape: {roi_arr.shape}")
    if roi_arr.ndim == 4 and roi_arr.shape[1] not in (1, 3):
        raise ValueError(f"Unexpected channel dimension in ROI images: {roi_arr.shape}")

    total_rois = int(roi_arr.shape[0])
    meta = {
        "eye_masks_run": run_name,
        "eye_masks_method": run_grp.attrs.get("method"),
        "crop_run": crop_run,
        "use_probs": use_probs,
        "targets_source": target_source,
        "length": total_rois,
        "roi_shape": tuple(int(s) for s in roi_arr.shape[1:]),
        "target_channels": 1 if label_mode == "union" else 2,
    }
    eye_labels_attr = run_grp.attrs.get("eye_labels")
    if isinstance(eye_labels_attr, (list, tuple)):
        meta["eye_labels"] = [str(label) for label in eye_labels_attr]
    elif isinstance(eye_labels_attr, str):
        meta["eye_labels"] = [eye_labels_attr]
    if use_probs and mask_probs_arr is not None:
        meta["mask_probs_dtype"] = str(mask_probs_arr.dtype)
    if not use_probs and masks_arr is not None:
        meta["masks_dtype"] = str(masks_arr.dtype)

    return YOLOTargetStore(
        roi_array=roi_arr,
        mask_probs_array=mask_probs_arr,
        masks_array=masks_arr,
        use_probs=use_probs,
        meta=meta,
    )


def _cache_resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    if target_size and (image.shape[0] != target_size or image.shape[1] != target_size):
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image


def _cache_resize_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    if target_size and (mask.shape[0] != target_size or mask.shape[1] != target_size):
        resized = cv2.resize(mask.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)
    return mask


def _cache_resize_mask_stack(mask_stack: np.ndarray) -> np.ndarray:
    """Resize a stack of binary masks (C, H, W) to the cache target size."""
    channels, height, width = mask_stack.shape
    if _CACHE_TARGET_SIZE and (height != _CACHE_TARGET_SIZE or width != _CACHE_TARGET_SIZE):
        resized = np.zeros((channels, _CACHE_TARGET_SIZE, _CACHE_TARGET_SIZE), dtype=np.uint8)
        for ch in range(channels):
            resized[ch] = (
                cv2.resize(
                    mask_stack[ch].astype(np.uint8),
                    (_CACHE_TARGET_SIZE, _CACHE_TARGET_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            ).astype(np.uint8)
        return resized
    return mask_stack


def _cache_process_initializer(
    images_path: str,
    masks_path: str,
    length: int,
    target_size: int,
    mask_channels: int,
    context_map: Dict[str, Dict[str, Optional[str]]],
) -> None:
    global _CACHE_PROCESS_IMAGES, _CACHE_PROCESS_MASKS, _CACHE_TARGET_SIZE, _CACHE_MASK_CHANNELS, _CACHE_CONTEXTS
    _CACHE_TARGET_SIZE = target_size
    _CACHE_MASK_CHANNELS = max(1, int(mask_channels))
    _CACHE_PROCESS_IMAGES = np.memmap(
        images_path,
        dtype=np.uint8,
        mode="r+",
        shape=(length, target_size, target_size),
    )
    _CACHE_PROCESS_MASKS = np.memmap(
        masks_path,
        dtype=np.uint8,
        mode="r+",
        shape=(length, _CACHE_MASK_CHANNELS, target_size, target_size),
    )
    _CACHE_CONTEXTS = {}
    for name, ctx in context_map.items():
        root = zarr.open(ctx["zarr_path"], mode="r")
        cache_entry = {
            "roi_images": root[f"crop_runs/{ctx['crop_run']}/roi_images"],
            "masks": root[f"eye_masks_runs/{ctx['mask_run']}/masks_roi"],
            "roi_coords": root[f"crop_runs/{ctx['crop_run']}/roi_coordinates_full"],
            "background": None,
        }
        if ctx.get("include_background") and ctx.get("background_run") and ctx.get("background_array"):
            cache_entry["background"] = root[
                f"background_runs/{ctx['background_run']}/{ctx['background_array']}"
            ]
        _CACHE_CONTEXTS[name] = cache_entry


def _cache_process_worker(task: Dict[str, object]) -> int:
    idx = int(task["index"])
    dataset_name = str(task["dataset_name"])
    entry_type = str(task["entry_type"])
    roi_index = int(task["roi_index"])
    roi_shape_raw = task["roi_shape"]
    roi_shape = tuple(int(v) for v in roi_shape_raw)  # type: ignore[arg-type]

    context = _CACHE_CONTEXTS[dataset_name]

    if entry_type == "background":
        roi_coords = context["roi_coords"][roi_index]
        x1, y1 = map(int, roi_coords)
        roi_h, roi_w = roi_shape
        x2, y2 = x1 + roi_w, y1 + roi_h
        background_arr = context["background"]
        if background_arr is None:
            raise ValueError("Background array missing for background cache entry")
        background_roi = np.asarray(background_arr[y1:y2, x1:x2])
        if background_roi.shape != roi_shape:
            background_roi = cv2.resize(
                background_roi,
                (roi_w, roi_h),
                interpolation=cv2.INTER_LINEAR,
            )
        image_source = background_roi
        mask_stack = np.zeros((_CACHE_MASK_CHANNELS, roi_h, roi_w), dtype=np.uint8)
    else:
        roi_img = np.asarray(context["roi_images"][roi_index])
        raw_masks = np.asarray(context["masks"][roi_index]).astype(np.uint8)
        if raw_masks.ndim == 2:
            raw_masks = raw_masks[None, ...]
        channels = min(_CACHE_MASK_CHANNELS, raw_masks.shape[0])
        mask_stack = np.zeros((_CACHE_MASK_CHANNELS, raw_masks.shape[1], raw_masks.shape[2]), dtype=np.uint8)
        if channels > 0:
            mask_stack[:channels] = raw_masks[:channels]
        image_source = roi_img

    image_resized = _cache_resize_image(image_source, _CACHE_TARGET_SIZE)
    mask_resized_stack = _cache_resize_mask_stack(mask_stack)

    _CACHE_PROCESS_IMAGES[idx, :, :] = image_resized  # type: ignore[index]
    _CACHE_PROCESS_MASKS[idx, :, :, :] = mask_resized_stack  # type: ignore[index]
    return idx
class EyeMaskYOLODataset(Dataset):
    """PyTorch dataset that yields samples compatible with Ultralytics segmentation trainer."""

    def __init__(
        self,
        entries: Sequence[_EyeMaskEntry],
        target_size: int,
        cache_config: Optional[EyeMaskCacheConfig] = None,
        split_name: str = "train",
        console: Optional[Console] = None,
        dataset_cache_info: Optional[Dict[str, DatasetCacheInfo]] = None,
        mask_smoothing_kernel: int = 0,
        edge_enhancement: Optional[str] = None,
        binarization_threshold: Optional[int] = None,
    ) -> None:
        self.entries = list(entries)
        self.target_size = target_size
        self.cache_config = cache_config
        self.split_name = split_name
        self.console = console
        self.dataset_cache_info = dataset_cache_info or {}
        self.cached_images: Optional[np.memmap] = None
        self.cached_masks: Optional[np.memmap] = None
        self.cached_mask_channels: Optional[int] = None
        kernel = int(mask_smoothing_kernel or 0)
        if kernel < 0:
            kernel = 0
        if kernel > 0 and kernel % 2 == 0:
            kernel += 1  # ensure odd for morphological ops
        self.mask_smoothing_kernel = kernel
        self.edge_enhancement = (edge_enhancement or "").strip().lower() or None
        if binarization_threshold is not None and not (0 <= int(binarization_threshold) <= 255):
            raise ValueError("binarization_threshold must be between 0 and 255")
        self.binarization_threshold = None if binarization_threshold is None else int(binarization_threshold)
        self.mask_channels = max(1, self._infer_mask_channels())
        if self.cache_config and self.cache_config.enabled:
            self._prepare_cache()

    def __len__(self) -> int:
        return len(self.entries)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.target_size and (image.shape[0] != self.target_size or image.shape[1] != self.target_size):
            return cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        return image

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.target_size and (mask.shape[0] != self.target_size or mask.shape[1] != self.target_size):
            resized = cv2.resize(mask.astype(np.uint8), (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            return (resized > 0).astype(np.uint8)
        return mask

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.mask_smoothing_kernel <= 0:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.mask_smoothing_kernel, self.mask_smoothing_kernel))
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return (opened > 0).astype(np.uint8)

    def _edge_channel(self, image: np.ndarray) -> np.ndarray:
        if not self.edge_enhancement:
            return image

        method = self.edge_enhancement
        image_float = image.astype(np.float32)

        if method == "sobel":
            grad_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(grad_x, grad_y)
        elif method == "laplacian":
            magnitude = cv2.Laplacian(image_float, cv2.CV_32F, ksize=3)
            magnitude = np.abs(magnitude)
        elif method == "canny":
            magnitude = cv2.Canny(image.astype(np.uint8), 50, 150).astype(np.float32)
        else:
            return image

        max_val = float(magnitude.max())
        if max_val <= 0:
            return np.zeros_like(image, dtype=np.uint8)
        normalized = (magnitude / max_val) * 255.0
        return normalized.astype(np.uint8)

    def _compose_image_tensor(self, grayscale: np.ndarray) -> np.ndarray:
        channels: List[np.ndarray] = []

        base_image = grayscale
        channels.append(base_image)

        if self.binarization_threshold is not None:
            binary = (base_image > self.binarization_threshold).astype(np.uint8) * 255
            channels.append(binary)

        if self.edge_enhancement:
            edge_channel = self._edge_channel(base_image)
            channels.append(edge_channel)

        while len(channels) < 3:
            channels.append(base_image)

        stacked = np.stack(channels[:3], axis=0)
        return stacked.astype(np.float32)

    def _mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        x_center = x_min + w / 2.0
        y_center = y_min + h / 2.0
        width = mask.shape[1]
        height = mask.shape[0]
        bbox = np.array([[x_center / width, y_center / height, w / width, h / height]], dtype=np.float32)
        return bbox

    def _mask_to_segments(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments: List[np.ndarray] = []
        if not contours:
            return segments
        h, w = mask.shape
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            flattened = cnt.reshape(-1, 2).astype(np.float32)
            flattened[:, 0] /= float(w)
            flattened[:, 1] /= float(h)
            segments.append(flattened)
        return segments

    def _compute_cache_signature(self) -> str:
        h = hashlib.sha1()
        h.update(str(self.target_size).encode())
        h.update(str(len(self.entries)).encode())
        h.update(str(self.mask_smoothing_kernel).encode())
        h.update((self.edge_enhancement or "none").encode())
        h.update(str(self.binarization_threshold if self.binarization_threshold is not None else "none").encode())
        for entry in self.entries:
            h.update(entry.dataset_name.encode())
            h.update(entry.zarr_path.encode())
            h.update(entry.crop_run.encode())
            if entry.mask_run:
                h.update(entry.mask_run.encode())
            h.update(str(entry.roi_index).encode())
            h.update(entry.entry_type.encode())
            h.update(str(int(entry.has_positive)).encode())
        return h.hexdigest()

    def _infer_mask_channels(self) -> int:
        for entry in self.entries:
            if entry.entry_type == "roi" and entry.masks is not None:
                mask = np.asarray(entry.masks[entry.roi_index])
                if mask.ndim == 2:
                    return 1
                return int(mask.shape[0])
        return 1

    def _prepare_cache(self) -> None:
        if self.console:
            self.console.log(f"[green]EyeMask caching enabled[/green] • [yellow]{self.split_name}[/yellow] samples: {len(self.entries):,}")
        signature = self._compute_cache_signature()
        cache_root = self.cache_config.directory / self.split_name
        cache_root.mkdir(parents=True, exist_ok=True)
        stem = f"{signature[:12]}_{len(self.entries)}_{self.target_size}"
        meta_path = cache_root / f"{stem}.json"
        images_path = cache_root / f"{stem}_images.dat"
        masks_path = cache_root / f"{stem}_masks.dat"

        expected_mask_channels = max(1, self.mask_channels)
        metadata_matches = False
        cached_mask_channels = None
        if self.cache_config.reuse_existing and meta_path.exists() and images_path.exists() and masks_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                cached_mask_channels = meta.get("mask_channels")
                metadata_matches = (
                    meta.get("signature") == signature
                    and meta.get("length") == len(self.entries)
                    and meta.get("target_size") == self.target_size
                    and cached_mask_channels == expected_mask_channels
                )
            except (json.JSONDecodeError, OSError):
                metadata_matches = False

        if not metadata_matches:
            if self.console:
                self.console.log(
                    f"[yellow]Building cache[/yellow] • [yellow]{self.split_name}[/yellow] ({len(self.entries):,} samples)"
                )
            if images_path.exists():
                images_path.unlink()
            if masks_path.exists():
                masks_path.unlink()
            image_map = np.memmap(
                images_path,
                dtype=np.uint8,
                mode="w+",
                shape=(len(self.entries), self.target_size, self.target_size),
            )
            mask_map = np.memmap(
                masks_path,
                dtype=np.uint8,
                mode="w+",
                shape=(len(self.entries), expected_mask_channels, self.target_size, self.target_size),
            )
            total_entries = len(self.entries)
            backend = CacheBackend.THREAD
            worker_count = 1
            if self.cache_config:
                backend = self.cache_config.backend
                worker_count = max(1, min(self.cache_config.workers, total_entries))

            progress_ctx = nullcontext()
            progress = None
            task_id = None
            if self.console and total_entries > 0:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeRemainingColumn(),
                    console=self.console,
                    transient=True,
                )
                progress_ctx = progress

            with progress_ctx:
                if progress:
                    task_id = progress.add_task(
                        f"[cyan]Caching {self.split_name}[/cyan]", total=total_entries
                    )

                if backend == CacheBackend.PROCESS and worker_count > 1 and total_entries > 0:
                    context_map = {
                        name: info.to_dict() for name, info in self.dataset_cache_info.items()
                    }
                    tasks = [
                        {
                            "index": idx,
                            "dataset_name": entry.dataset_name,
                            "entry_type": entry.entry_type,
                            "roi_index": entry.roi_index,
                            "roi_shape": entry.roi_shape,
                        }
                        for idx, entry in enumerate(self.entries)
                    ]
                    initargs = (
                        str(images_path),
                        str(masks_path),
                        total_entries,
                        self.target_size,
                        expected_mask_channels,
                        context_map,
                    )
                    with ProcessPoolExecutor(
                        max_workers=worker_count,
                        initializer=_cache_process_initializer,
                        initargs=initargs,
                    ) as executor:
                        for _ in executor.map(_cache_process_worker, tasks, chunksize=32):
                            if progress and task_id is not None:
                                progress.update(task_id, advance=1)
                else:
                    def process_entry(item: Tuple[int, _EyeMaskEntry]) -> Tuple[int, np.ndarray, np.ndarray]:
                        idx, entry = item
                        image_resized, _ = self._load_resized_arrays(entry)
                        mask_stack = self._load_mask_channels_from_source(
                            entry,
                            mask_channels=expected_mask_channels,
                            apply_resize=True,
                        )
                        return idx, image_resized, mask_stack

                    if worker_count > 1 and total_entries > 0:
                        with ThreadPoolExecutor(max_workers=worker_count) as executor:
                            for idx, image_resized, mask_stack in executor.map(
                                process_entry, enumerate(self.entries), chunksize=32
                            ):
                                image_map[idx, :, :] = image_resized
                                mask_map[idx, :, :, :] = mask_stack
                                if progress and task_id is not None:
                                    progress.update(task_id, advance=1)
                    else:
                        for idx, entry in enumerate(self.entries):
                            image_resized, _ = self._load_resized_arrays(entry)
                            mask_stack = self._load_mask_channels_from_source(
                                entry,
                                mask_channels=expected_mask_channels,
                                apply_resize=True,
                            )
                            image_map[idx, :, :] = image_resized
                            mask_map[idx, :, :, :] = mask_stack
                            if progress and task_id is not None:
                                progress.update(task_id, advance=1)
            image_map.flush()
            mask_map.flush()
            del image_map
            del mask_map
            metadata = {
                "signature": signature,
                "length": len(self.entries),
                "target_size": self.target_size,
                "images_path": str(images_path),
                "masks_path": str(masks_path),
                "mask_channels": expected_mask_channels,
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            if self.console:
                self.console.log(f"[green]Cache built[/green] • [yellow]{self.split_name}[/yellow]")
        else:
            expected_mask_channels = int(cached_mask_channels)
            if self.console:
                self.console.log(
                    f"[green]Reusing cache[/green] • [yellow]{self.split_name}[/yellow] ({stem})"
                )
        self.cached_images = np.memmap(
            images_path,
            dtype=np.uint8,
            mode="r",
            shape=(len(self.entries), self.target_size, self.target_size),
        )
        self.cached_masks = np.memmap(
            masks_path,
            dtype=np.uint8,
            mode="r",
            shape=(len(self.entries), expected_mask_channels, self.target_size, self.target_size),
        )
        self.cached_mask_channels = expected_mask_channels
        self.mask_channels = expected_mask_channels
        self.cached_mask_channels = expected_mask_channels

    def _load_resized_arrays(self, entry: _EyeMaskEntry) -> Tuple[np.ndarray, np.ndarray]:
        if entry.entry_type == "background":
            if entry.background_full is None or entry.roi_coordinates_full is None:
                raise ValueError("Background entry missing background data")
            x1, y1 = map(int, entry.roi_coordinates_full[entry.roi_index])
            roi_h, roi_w = entry.roi_shape
            x2, y2 = x1 + roi_w, y1 + roi_h
            background_roi = entry.background_full[y1:y2, x1:x2]
            if background_roi.shape != entry.roi_shape:
                background_roi = cv2.resize(
                    background_roi,
                    (roi_w, roi_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            combined_mask = np.zeros(entry.roi_shape, dtype=np.uint8)
            image_source = background_roi
        else:
            if entry.roi_images is None or entry.masks is None:
                raise ValueError("Positive entry missing ROI data")
            roi_img = np.asarray(entry.roi_images[entry.roi_index])
            roi_mask_lr = np.asarray(entry.masks[entry.roi_index])
            combined_mask = np.logical_or(roi_mask_lr[0] > 0, roi_mask_lr[1] > 0).astype(np.uint8)
            image_source = roi_img

        image_resized = self._resize_image(image_source)
        mask_resized = self._resize_mask(combined_mask)
        mask_resized = self._smooth_mask(mask_resized)
        return image_resized, mask_resized

    def _load_mask_channels_from_source(
        self,
        entry: _EyeMaskEntry,
        mask_channels: Optional[int] = None,
        apply_resize: bool = True,
    ) -> np.ndarray:
        """Load per-eye mask channels for an entry, optionally resized to target size."""
        target_channels = max(1, int(mask_channels or self.mask_channels))
        if entry.entry_type != "roi" or entry.masks is None:
            height = self.target_size if apply_resize and self.target_size else entry.roi_shape[0]
            width = self.target_size if apply_resize and self.target_size else entry.roi_shape[1]
            return np.zeros((target_channels, height, width), dtype=np.uint8)

        mask_lr = np.asarray(entry.masks[entry.roi_index]).astype(np.uint8)
        if mask_lr.ndim == 2:
            mask_lr = mask_lr[None, ...]

        if apply_resize and self.target_size and (
            mask_lr.shape[1] != self.target_size or mask_lr.shape[2] != self.target_size
        ):
            resized = np.zeros((mask_lr.shape[0], self.target_size, self.target_size), dtype=np.uint8)
            for ch in range(mask_lr.shape[0]):
                resized[ch] = self._resize_mask(mask_lr[ch])
            mask_lr = resized

        channels = min(target_channels, mask_lr.shape[0])
        result = np.zeros((target_channels, mask_lr.shape[1], mask_lr.shape[2]), dtype=np.uint8)
        if channels > 0:
            result[:channels] = mask_lr[:channels]
        return result

    def _apply_mask_smoothing(self, mask_stack: np.ndarray) -> np.ndarray:
        if self.mask_smoothing_kernel <= 0:
            return mask_stack.astype(np.uint8, copy=False)
        smoothed = np.zeros_like(mask_stack, dtype=np.uint8)
        for idx, mask_ch in enumerate(mask_stack):
            smoothed[idx] = self._smooth_mask(mask_ch).astype(np.uint8)
        return smoothed

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]
        if self.cached_images is not None and self.cached_masks is not None:
            image_resized = np.asarray(self.cached_images[idx])
            mask_stack_raw = np.asarray(self.cached_masks[idx])
            mask_resized = (mask_stack_raw > 0).any(axis=0).astype(np.uint8)
        else:
            image_resized, mask_resized = self._load_resized_arrays(entry)
            mask_stack_raw = self._load_mask_channels_from_source(entry, mask_channels=self.mask_channels, apply_resize=True)

        image_tensor = self._compose_image_tensor(image_resized)

        bboxes_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        segments: List[np.ndarray] = []
        label_indices: List[int] = []

        if entry.entry_type == "roi":
            mask_stack = self._apply_mask_smoothing(mask_stack_raw)
            for ch_idx, mask_ch in enumerate(mask_stack):
                mask_bool = mask_ch > 0
                if mask_bool.sum() == 0:
                    continue
                bbox = self._mask_to_bbox(mask_bool)
                if bbox.size == 0:
                    continue
                bboxes_list.append(bbox[0])
                mask_list.append(mask_bool.astype(np.float32))
                label_indices.append(ch_idx)
                seg_candidates = self._mask_to_segments(mask_bool)
                if seg_candidates:
                    best_seg = max(seg_candidates, key=lambda arr: arr.shape[0])
                    segments.append(best_seg)
                else:
                    segments.append(np.zeros((0, 2), dtype=np.float32))

        if mask_list:
            mask_tensor = np.stack(mask_list, axis=0).astype(np.float32)
            bboxes = np.asarray(bboxes_list, dtype=np.float32)
            cls = np.zeros((mask_tensor.shape[0],), dtype=np.float32)
        else:
            height, width = mask_resized.shape
            mask_tensor = np.zeros((0, height, width), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
            segments = []
            label_indices = []

        suffix = "background" if entry.entry_type == "background" else "roi"
        im_file = f"{Path(entry.zarr_path).stem}_{suffix}_{entry.roi_index}"

        sample = {
            "img": image_tensor,
            "cls": cls,
            "bboxes": bboxes,
            "masks": mask_tensor,
            "segments": segments,
            "im_file": im_file,
            "ori_shape": (mask_resized.shape[0], mask_resized.shape[1]),
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
            "mask_labels": np.asarray(label_indices, dtype=np.int32),
        }
        return sample


@dataclass
class EyeMaskDatasetBundle:
    train_dataset: EyeMaskYOLODataset
    val_dataset: EyeMaskYOLODataset
    stats: Dict[str, EyeMaskDatasetStats]
    cache_info: Dict[str, DatasetCacheInfo]


def _resolve_runs(root: zarr.Group, cfg: EyeMaskDatasetConfig) -> Tuple[str, str]:
    crop_parent = root.get("crop_runs")
    if crop_parent is None or "latest" not in crop_parent.attrs:
        raise ValueError(f"Zarr store '{root.store.path}' is missing crop_runs/latest")
    crop_run = cfg.crop_run or crop_parent.attrs["latest"]
    if crop_run not in crop_parent:
        raise ValueError(f"Crop run '{crop_run}' not found in {root.store.path}")

    mask_parent = root.get("eye_masks_runs")
    if mask_parent is None or "latest" not in mask_parent.attrs:
        raise ValueError(f"Zarr store '{root.store.path}' is missing eye_masks_runs/latest")
    mask_run = cfg.mask_run or mask_parent.attrs["latest"]
    if mask_run not in mask_parent:
        raise ValueError(f"Eye mask run '{mask_run}' not found in {root.store.path}")

    return crop_run, mask_run


def _build_entries_for_dataset(
    dataset_name: str,
    cfg: EyeMaskDatasetConfig,
    default_split: DatasetSplit,
    rng: np.random.Generator,
    console: Optional[Console] = None,
) -> Tuple[List[_EyeMaskEntry], List[_EyeMaskEntry], EyeMaskDatasetStats, DatasetCacheInfo]:
    root = zarr.open(str(cfg.zarr_path), mode="r")
    crop_run, mask_run = _resolve_runs(root, cfg)

    roi_images = root[f"crop_runs/{crop_run}/roi_images"]
    roi_coords_full = root[f"crop_runs/{crop_run}/roi_coordinates_full"][:]
    roi_shape = (int(roi_images.shape[1]), int(roi_images.shape[2]))
    mask_group = root[f"eye_masks_runs/{mask_run}"]
    if "ellipse_success" not in mask_group:
        raise ValueError(f"Eye mask run '{mask_run}' missing 'ellipse_success' array")
    masks = mask_group["masks_roi"]
    ellipse_success = mask_group["ellipse_success"][:].astype(bool)
    total_rois = roi_images.shape[0]
    if masks.shape[0] != total_rois:
        raise ValueError(
            f"Mismatch between roi_images ({total_rois}) and masks ({masks.shape[0]}) in {cfg.zarr_path}"
        )

    background_full: Optional[np.ndarray] = None
    bg_run_name: Optional[str] = None
    array_name: Optional[str] = None
    if cfg.include_background_negatives:
        bg_parent = root.get("background_runs")
        if bg_parent is None or "latest" not in bg_parent.attrs:
            raise ValueError(
                f"Zarr store '{cfg.zarr_path}' is missing background runs required for background negatives"
            )
        bg_run_name = cfg.background_run or bg_parent.attrs["latest"]
        bg_group = root[f"background_runs/{bg_run_name}"]
        array_name = "background_ds" if cfg.background_from_downsampled else "background_full"
        if array_name not in bg_group:
            raise ValueError(f"Background run '{bg_run_name}' missing '{array_name}' array")
        background_full = np.asarray(bg_group[array_name][:])

    indices = np.arange(total_rois, dtype=np.int32)
    rng.shuffle(indices)

    split = cfg.split or default_split
    train_count = int(round(split.train * total_rois))
    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    positive_count = 0
    negative_count = 0
    background_count = 0
    skipped_empty_due_to_filter = 0
    filter_reasons: Counter[str] = Counter()
    dataset_start = time.perf_counter()

    if console:
        console.log(
            f"[yellow]{dataset_name}[/yellow] • total_rois={total_rois:,} • "
            f"target_split=train:{len(train_indices):,}, val:{len(val_indices):,} • "
            f"require_both_eyes={cfg.require_both_eyes} • "
            f"using_ellipse_success_filter=True • "
            f"include_empty={cfg.include_empty} • "
            f"include_background_negatives={cfg.include_background_negatives} • "
            f"background_ratio={cfg.background_negative_ratio}"
        )

    def build_list(split_name: str, source_indices: Sequence[int]) -> Tuple[List[_EyeMaskEntry], List[int]]:
        nonlocal positive_count
        nonlocal negative_count
        nonlocal skipped_empty_due_to_filter
        processed = 0
        split_total = len(source_indices)
        progress_step = max(100, split_total // 10) if split_total else 0
        local_positive = 0
        local_negative = 0
        local_skipped = 0
        entries: List[_EyeMaskEntry] = []
        positive_indices: List[int] = []
        for idx in source_indices:
            success_flags = ellipse_success[idx]
            left_success = bool(success_flags[0])
            right_success = bool(success_flags[1])
            if cfg.require_both_eyes:
                has_positive = left_success and right_success
            else:
                has_positive = left_success or right_success

            if has_positive:
                positive_count += 1  # type: ignore[misc]
                local_positive += 1
                positive_indices.append(int(idx))
            else:
                reasons: List[str] = []
                if not left_success and not right_success:
                    reasons.append("both_fail")
                else:
                    if not left_success:
                        reasons.append("left_fail")
                    if not right_success:
                        reasons.append("right_fail")
                if not reasons:
                    reasons.append("unknown_fail")
                for reason in reasons:
                    filter_reasons[reason] += 1
                if not cfg.include_empty:
                    skipped_empty_due_to_filter += 1
                    local_skipped += 1
                    continue
                negative_count += 1  # type: ignore[misc]
                local_negative += 1
            entries.append(
                _EyeMaskEntry(
                    dataset_name=dataset_name,
                    zarr_path=str(cfg.zarr_path),
                    crop_run=crop_run,
                    mask_run=mask_run,
                    roi_images=roi_images,
                    masks=masks,
                    roi_coordinates_full=None,
                    background_full=None,
                    roi_index=int(idx),
                    roi_shape=roi_shape,
                    has_positive=has_positive,
                    entry_type="roi",
                )
            )
            processed += 1
            if console and progress_step and processed % progress_step == 0:
                console.log(
                    f"[yellow]{dataset_name}[/yellow] • {split_name} processed {processed:,}/{split_total:,} "
                    f"(positives={local_positive:,}, negatives={local_negative:,}, skipped={local_skipped:,})"
                )
        if console and split_total:
            console.log(
                f"[yellow]{dataset_name}[/yellow] • {split_name} completed "
                f"{processed:,}/{split_total:,} (positives={local_positive:,}, negatives={local_negative:,}, skipped={local_skipped:,})"
            )
        return entries, positive_indices

    train_entries, train_positive_indices = build_list("train", train_indices)
    val_entries, val_positive_indices = build_list("val", val_indices)

    def add_background_entries(indices: Sequence[int], destination: List[_EyeMaskEntry]) -> None:
        nonlocal background_count
        if background_full is None or not cfg.include_background_negatives:
            return
        if not indices:
            return
        ratio = cfg.background_negative_ratio
        target = max(1, int(round(len(indices) * ratio)))
        target = min(target, len(indices))
        selection = (
            indices if target >= len(indices) else rng.choice(indices, size=target, replace=False)
        )
        for idx in selection:
            destination.append(
                _EyeMaskEntry(
                    dataset_name=dataset_name,
                    zarr_path=str(cfg.zarr_path),
                    crop_run=crop_run,
                    mask_run=mask_run,
                    roi_images=None,
                    masks=None,
                    roi_coordinates_full=roi_coords_full,
                    background_full=background_full,
                    roi_index=int(idx),
                    roi_shape=roi_shape,
                    has_positive=False,
                    entry_type="background",
                )
            )
            background_count += 1  # type: ignore[misc]

    add_background_entries(train_positive_indices, train_entries)
    add_background_entries(val_positive_indices, val_entries)

    stats = EyeMaskDatasetStats(
        dataset_name=dataset_name,
        zarr_path=str(cfg.zarr_path),
        crop_run=crop_run,
        mask_run=mask_run,
        total_rois=total_rois,
        positive_rois=positive_count,
        negative_rois=negative_count,
        background_negatives=background_count,
    )

    if console:
        dataset_duration = time.perf_counter() - dataset_start
        console.log(
            f"[yellow]{dataset_name}[/yellow] • positives={positive_count:,} "
            f"negatives_included={negative_count:,} • "
            f"background_negatives={background_count:,} • "
            f"skipped_empty={skipped_empty_due_to_filter:,} • "
            f"build_time={dataset_duration:.2f}s"
        )
        if filter_reasons:
            reason_summary = ", ".join(f"{reason}={count}" for reason, count in filter_reasons.most_common())
            console.log(f"[yellow]{dataset_name}[/yellow] • filter_reasons: {reason_summary}")
        console.log(
            f"[yellow]{dataset_name}[/yellow] • train_samples={len(train_entries):,} • val_samples={len(val_entries):,}"
        )
    cache_info = DatasetCacheInfo(
        dataset_name=dataset_name,
        zarr_path=str(cfg.zarr_path),
        crop_run=crop_run,
        mask_run=mask_run,
        include_background=cfg.include_background_negatives,
        background_run=bg_run_name if cfg.include_background_negatives else None,
        background_array=array_name if cfg.include_background_negatives else None,
    )

    return train_entries, val_entries, stats, cache_info


def build_eye_mask_datasets(
    config: EyeMaskTrainingConfig,
    console: Optional[Console] = None,
) -> EyeMaskDatasetBundle:
    """Create train/val datasets and gather basic statistics."""
    rng = np.random.default_rng(config.random_seed)

    all_train_entries: List[_EyeMaskEntry] = []
    all_val_entries: List[_EyeMaskEntry] = []
    stats: Dict[str, EyeMaskDatasetStats] = {}
    cache_info_map: Dict[str, DatasetCacheInfo] = {}

    for dataset_name, dataset_cfg in config.datasets.items():
        train_entries, val_entries, dataset_stats, cache_info = _build_entries_for_dataset(
            dataset_name=dataset_name,
            cfg=dataset_cfg,
            default_split=config.default_split,
            rng=rng,
            console=console,
        )
        if not train_entries:
            raise ValueError(f"No training samples available for dataset '{dataset_name}'")
        if not val_entries:
            raise ValueError(f"No validation samples available for dataset '{dataset_name}'")
        all_train_entries.extend(train_entries)
        all_val_entries.extend(val_entries)
        stats[dataset_name] = dataset_stats
        cache_info_map[dataset_name] = cache_info

    cache_cfg = config.cache
    train_dataset = EyeMaskYOLODataset(
        all_train_entries,
        target_size=config.target_size,
        cache_config=cache_cfg,
        split_name="train",
        console=console,
        dataset_cache_info=cache_info_map,
        mask_smoothing_kernel=config.mask_smoothing_kernel,
        edge_enhancement=config.edge_enhancement,
        binarization_threshold=config.binarization_threshold,
    )
    val_dataset = EyeMaskYOLODataset(
        all_val_entries,
        target_size=config.target_size,
        cache_config=cache_cfg,
        split_name="val",
        console=console,
        dataset_cache_info=cache_info_map,
        mask_smoothing_kernel=config.mask_smoothing_kernel,
        edge_enhancement=config.edge_enhancement,
        binarization_threshold=config.binarization_threshold,
    )

    return EyeMaskDatasetBundle(train_dataset=train_dataset, val_dataset=val_dataset, stats=stats, cache_info=cache_info_map)
