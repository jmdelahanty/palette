"""Zarr-backed dataset helpers for unified subject-mask U-Net training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from rich.console import Console
from torch.utils.data import Dataset

from .config import SubjectMaskDatasetConfig, SubjectMaskTrainingConfig


@dataclass
class SubjectMaskTargetStore:
    roi_array: zarr.Array
    masks_array: zarr.Array
    valid_array: zarr.Array
    train_indices: np.ndarray
    val_indices: np.ndarray
    meta: Dict[str, object]


@dataclass
class SubjectMaskDatasetBundle:
    train_dataset: "SubjectMaskChunkedDataset"
    val_dataset: "SubjectMaskChunkedDataset"
    meta_list: List[Dict[str, object]]
    label_schema_id: str
    mask_labels: Tuple[str, ...]


def _sorted_group_keys(group: Optional[zarr.Group]) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    return sorted(key for key in keys if isinstance(key, str))


def _resolve_run(
    parent: zarr.Group | None,
    *,
    requested: Optional[str],
    latest_attr: str = "latest",
    context: str,
) -> Tuple[str, zarr.Group]:
    if parent is None:
        raise ValueError(f"Missing required group '{context}'.")
    if requested:
        if requested not in parent:
            raise ValueError(f"{context}/{requested} not found.")
        return str(requested), parent[str(requested)]
    latest = parent.attrs.get(latest_attr)
    if latest and str(latest) in parent:
        return str(latest), parent[str(latest)]
    names = _sorted_group_keys(parent)
    if not names:
        raise ValueError(f"No runs found under '{context}'.")
    return names[-1], parent[names[-1]]


def _require_1d_indices(root: zarr.Group, name: str) -> np.ndarray:
    splits = root.get("splits")
    if not isinstance(splits, zarr.Group) or name not in splits:
        raise ValueError(f"Missing required split array splits/{name}.")
    values = np.asarray(splits[name][:], dtype=np.int64).reshape(-1)
    return values


def _normalize_roi_batch(batch: np.ndarray) -> np.ndarray:
    arr = np.asarray(batch)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    elif arr.ndim == 4 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.ndim == 4 and arr.shape[1] in (1, 3):
        pass
    else:
        raise ValueError(f"Unexpected ROI batch shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32, copy=False)
        if max_value > 0:
            arr /= max_value
    else:
        arr = arr.astype(np.float32, copy=False)
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val > 1.0:
            arr /= max_val
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def load_subject_mask_training_artifact(
    zarr_path: str | Path,
    *,
    subject_mask_run: Optional[str],
    crop_run: Optional[str],
    expected_label_schema_id: str,
) -> SubjectMaskTargetStore:
    """Resolve arrays for a merged subject-mask training artifact."""

    source_path = Path(zarr_path).expanduser().resolve()
    root = zarr.open_group(str(source_path), mode="r")

    training_task = str(root.attrs.get("training_task") or "").strip().lower()
    if training_task and training_task != "subject_masks":
        raise ValueError(f"{source_path}: expected training_task='subject_masks', got {training_task!r}.")

    crop_name, crop_group = _resolve_run(
        root.get("crop_runs"),
        requested=crop_run,
        context="crop_runs",
    )
    mask_name, mask_group = _resolve_run(
        root.get("subject_mask_runs"),
        requested=subject_mask_run,
        context="subject_mask_runs",
    )

    if "roi_images" not in crop_group:
        raise ValueError(f"{source_path}: crop_runs/{crop_name} missing roi_images.")
    if "masks_roi" not in mask_group or "target_valid_channels" not in mask_group:
        raise ValueError(
            f"{source_path}: subject_mask_runs/{mask_name} missing masks_roi or target_valid_channels."
        )

    label_schema_id = str(mask_group.attrs.get("label_schema_id") or "").strip()
    if label_schema_id != expected_label_schema_id:
        raise ValueError(
            f"{source_path}: expected label_schema_id={expected_label_schema_id!r}, got {label_schema_id!r}."
        )
    mask_labels_raw = mask_group.attrs.get("mask_labels")
    if not isinstance(mask_labels_raw, (list, tuple)):
        raise ValueError(f"{source_path}: subject_mask_runs/{mask_name} missing usable mask_labels attr.")
    mask_labels = tuple(str(item) for item in mask_labels_raw)

    roi_array = crop_group["roi_images"]
    masks_array = mask_group["masks_roi"]
    valid_array = mask_group["target_valid_channels"]
    if int(roi_array.shape[0]) != int(masks_array.shape[0]) or int(masks_array.shape[0]) != int(valid_array.shape[0]):
        raise ValueError(
            f"{source_path}: row mismatch between roi_images ({roi_array.shape[0]}), "
            f"masks_roi ({masks_array.shape[0]}), and target_valid_channels ({valid_array.shape[0]})."
        )
    if int(masks_array.shape[1]) != int(valid_array.shape[1]):
        raise ValueError(
            f"{source_path}: channel mismatch between masks_roi ({masks_array.shape[1]}) "
            f"and target_valid_channels ({valid_array.shape[1]})."
        )

    train_indices = _require_1d_indices(root, "train_indices")
    val_indices = _require_1d_indices(root, "val_indices")

    meta = {
        "zarr_path": str(source_path),
        "crop_run": crop_name,
        "subject_mask_run": mask_name,
        "label_schema_id": label_schema_id,
        "mask_labels": list(mask_labels),
        "length": int(roi_array.shape[0]),
        "roi_shape": tuple(int(v) for v in roi_array.shape[1:]),
        "target_shape": tuple(int(v) for v in masks_array.shape[1:]),
        "input_format": root.attrs.get("training_export", {}).get("input_format")
        if isinstance(root.attrs.get("training_export"), dict)
        else None,
        "channel_supervision_summary": (
            root.attrs.get("training_export", {}).get("channel_supervision_summary")
            if isinstance(root.attrs.get("training_export"), dict)
            else None
        ),
    }

    return SubjectMaskTargetStore(
        roi_array=roi_array,
        masks_array=masks_array,
        valid_array=valid_array,
        train_indices=train_indices,
        val_indices=val_indices,
        meta=meta,
    )


class _StoreChunkCache:
    """Lightweight reader that batches ROI/mask/valid loads per source chunk."""

    def __init__(self, store: SubjectMaskTargetStore) -> None:
        self.store = store
        total = int(store.roi_array.shape[0])
        chunk = getattr(store.roi_array, "chunks", (total,))[0] if hasattr(store.roi_array, "chunks") else total
        if not isinstance(chunk, int) or chunk <= 0:
            chunk = total
        self.chunk_size = max(1, chunk)
        self._current_chunk_id: Optional[int] = None
        self._roi_chunk: Optional[np.ndarray] = None
        self._mask_chunk: Optional[np.ndarray] = None
        self._valid_chunk: Optional[np.ndarray] = None

    def _load_chunk(self, chunk_id: int) -> None:
        start = chunk_id * self.chunk_size
        stop = min(start + self.chunk_size, int(self.store.roi_array.shape[0]))

        roi_np = _normalize_roi_batch(np.asarray(self.store.roi_array[start:stop]))
        mask_np = np.asarray(self.store.masks_array[start:stop], dtype=np.float32)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, None, :, :]
        if mask_np.ndim != 4:
            raise ValueError(f"Unexpected subject-mask chunk shape: {mask_np.shape}")
        max_val = float(np.nanmax(mask_np)) if mask_np.size else 0.0
        if max_val > 1.0:
            mask_np /= max_val
        mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=1.0, neginf=0.0)
        mask_np = np.clip(mask_np, 0.0, 1.0)

        valid_np = np.asarray(self.store.valid_array[start:stop], dtype=np.bool_)
        if valid_np.ndim != 2:
            raise ValueError(f"Unexpected target_valid_channels chunk shape: {valid_np.shape}")

        self._current_chunk_id = chunk_id
        self._roi_chunk = roi_np
        self._mask_chunk = mask_np
        self._valid_chunk = valid_np

    def get_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk_id = index // self.chunk_size
        if (
            self._current_chunk_id != chunk_id
            or self._roi_chunk is None
            or self._mask_chunk is None
            or self._valid_chunk is None
        ):
            self._load_chunk(chunk_id)
        offset = index - chunk_id * self.chunk_size
        return (
            self._roi_chunk[offset].copy(),
            self._mask_chunk[offset].copy(),
            self._valid_chunk[offset].copy(),
        )


class SubjectMaskChunkedDataset(Dataset):
    """Chunk-aware dataset for merged subject-mask training artifacts."""

    def __init__(
        self,
        entries: List[Tuple[SubjectMaskTargetStore, np.ndarray]],
        *,
        shuffle_chunks: bool,
        seed: int,
    ) -> None:
        self.records: List[Tuple[int, int]] = []
        self.groups: List[List[int]] = []
        self.caches: List[_StoreChunkCache] = []
        self.stores: List[SubjectMaskTargetStore] = []

        rng = np.random.default_rng(seed)
        for store, indices in entries:
            indices = np.asarray(indices, dtype=np.int64)
            if indices.size == 0:
                continue
            chunk_size = getattr(store.roi_array, "chunks", (int(store.roi_array.shape[0]),))[0]
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                chunk_size = int(store.roi_array.shape[0])

            chunk_map: Dict[int, List[int]] = {}
            for idx in indices:
                chunk_map.setdefault(int(idx // chunk_size), []).append(int(idx))
            chunk_ids = list(chunk_map.keys())
            if shuffle_chunks:
                rng.shuffle(chunk_ids)
            else:
                chunk_ids.sort()

            self.caches.append(_StoreChunkCache(store))
            self.stores.append(store)
            store_idx = len(self.stores) - 1

            for chunk_id in chunk_ids:
                chunk_indices = sorted(chunk_map[chunk_id])
                group_positions: List[int] = []
                for idx_val in chunk_indices:
                    self.records.append((store_idx, idx_val))
                    group_positions.append(len(self.records) - 1)
                self.groups.append(group_positions)

        if not self.records:
            raise ValueError("No samples available to build subject-mask dataset.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        store_idx, roi_idx = self.records[index]
        roi_np, mask_np, valid_np = self.caches[store_idx].get_sample(roi_idx)
        return {
            "img": roi_np,
            "masks": mask_np,
            "valid_channels": valid_np.astype(np.float32, copy=False),
        }


def build_subject_mask_training_datasets(
    config: SubjectMaskTrainingConfig,
    console: Optional[Console] = None,
) -> SubjectMaskDatasetBundle:
    """Create train/val datasets from merged subject-mask training artifacts."""

    expected_schema = str(config.training_params.label_schema_id)
    train_entries: List[Tuple[SubjectMaskTargetStore, np.ndarray]] = []
    val_entries: List[Tuple[SubjectMaskTargetStore, np.ndarray]] = []
    meta_list: List[Dict[str, object]] = []
    expected_labels: Optional[Tuple[str, ...]] = None

    for ds_idx, (name, ds_cfg) in enumerate(config.datasets.items()):
        store = load_subject_mask_training_artifact(
            ds_cfg.zarr_path,
            subject_mask_run=config.training_params.subject_masks_run or ds_cfg.subject_mask_run,
            crop_run=config.training_params.crop_run or ds_cfg.crop_run,
            expected_label_schema_id=expected_schema,
        )
        meta = dict(store.meta)
        meta["dataset_name"] = name
        meta["train_rows"] = int(store.train_indices.shape[0])
        meta["val_rows"] = int(store.val_indices.shape[0])
        meta_list.append(meta)

        labels = tuple(str(item) for item in store.meta["mask_labels"])
        if expected_labels is None:
            expected_labels = labels
        elif labels != expected_labels:
            raise ValueError(
                f"Dataset '{name}' mask_labels mismatch: {labels!r} != {expected_labels!r}."
            )

        if store.train_indices.size == 0:
            raise ValueError(f"Training split for dataset '{name}' is empty.")
        if store.val_indices.size == 0:
            raise ValueError(f"Validation split for dataset '{name}' is empty.")

        train_entries.append((store, store.train_indices))
        val_entries.append((store, store.val_indices))

        if console is not None:
            console.log(
                f"[yellow]{name}[/yellow] • train={store.train_indices.shape[0]:,} "
                f"val={store.val_indices.shape[0]:,} • schema={expected_schema}"
            )

    if expected_labels is None:
        raise ValueError("No datasets produced subject-mask training samples.")

    seed = int(config.random_seed)
    train_dataset = SubjectMaskChunkedDataset(train_entries, shuffle_chunks=True, seed=seed)
    val_dataset = SubjectMaskChunkedDataset(val_entries, shuffle_chunks=False, seed=seed + 10_000)
    return SubjectMaskDatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        meta_list=meta_list,
        label_schema_id=expected_schema,
        mask_labels=expected_labels,
    )
