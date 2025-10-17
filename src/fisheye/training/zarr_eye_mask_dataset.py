"""Zarr-backed dataset for eye-mask segmentation training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import zarr
from rich.console import Console
from torch.utils.data import Dataset

from .config import EyeMaskTrainingConfig, EyeMaskDatasetConfig, DatasetSplit


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


class EyeMaskYOLODataset(Dataset):
    """PyTorch dataset that yields samples compatible with Ultralytics segmentation trainer."""

    def __init__(
        self,
        entries: Sequence[_EyeMaskEntry],
        target_size: int,
    ) -> None:
        self.entries = list(entries)
        self.target_size = target_size

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

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]

        if entry.entry_type == "background":
            if entry.background_full is None or entry.roi_coordinates_full is None:
                raise ValueError("Background entry missing background data")
            x1, y1 = map(int, entry.roi_coordinates_full[entry.roi_index])
            roi_h, roi_w = entry.roi_shape
            x2, y2 = x1 + roi_w, y1 + roi_h
            background_roi = entry.background_full[y1:y2, x1:x2]
            if background_roi.shape != entry.roi_shape:
                # Fallback: resize to expected ROI shape
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

        image_rgb = np.stack([image_resized] * 3, axis=0).astype(np.float32)

        if entry.has_positive and mask_resized.sum() > 0:
            cls = np.array([0.0], dtype=np.float32)
            bboxes = self._mask_to_bbox(mask_resized)
            mask_tensor = mask_resized[None, ...].astype(np.float32)
            segments = self._mask_to_segments(mask_resized)
        else:
            cls = np.zeros((0,), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
            mask_tensor = np.zeros((0, mask_resized.shape[0], mask_resized.shape[1]), dtype=np.float32)
            segments = []

        suffix = "background" if entry.entry_type == "background" else "roi"
        im_file = f"{Path(entry.zarr_path).stem}_{suffix}_{entry.roi_index}"

        sample = {
            "img": image_rgb,
            "cls": cls,
            "bboxes": bboxes,
            "masks": mask_tensor,
            "segments": segments,
            "im_file": im_file,
            "ori_shape": (mask_resized.shape[0], mask_resized.shape[1]),
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        }
        return sample


@dataclass
class EyeMaskDatasetBundle:
    train_dataset: EyeMaskYOLODataset
    val_dataset: EyeMaskYOLODataset
    stats: Dict[str, EyeMaskDatasetStats]


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
) -> Tuple[List[_EyeMaskEntry], List[_EyeMaskEntry], EyeMaskDatasetStats]:
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
    return train_entries, val_entries, stats


def build_eye_mask_datasets(
    config: EyeMaskTrainingConfig,
    console: Optional[Console] = None,
) -> EyeMaskDatasetBundle:
    """Create train/val datasets and gather basic statistics."""
    rng = np.random.default_rng(config.random_seed)

    all_train_entries: List[_EyeMaskEntry] = []
    all_val_entries: List[_EyeMaskEntry] = []
    stats: Dict[str, EyeMaskDatasetStats] = {}

    for dataset_name, dataset_cfg in config.datasets.items():
        train_entries, val_entries, dataset_stats = _build_entries_for_dataset(
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

    train_dataset = EyeMaskYOLODataset(all_train_entries, target_size=config.target_size)
    val_dataset = EyeMaskYOLODataset(all_val_entries, target_size=config.target_size)

    return EyeMaskDatasetBundle(train_dataset=train_dataset, val_dataset=val_dataset, stats=stats)
