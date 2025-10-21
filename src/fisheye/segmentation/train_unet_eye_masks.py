"""Train a lightweight U-Net to predict eye-mask probabilities stored in Zarr."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import os
import shutil
import numpy as np
import torch
import torch._dynamo
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TextColumn,
    SpinnerColumn,
)
from torch.utils.data import DataLoader, Dataset, Sampler

from ..training.config import EyeMaskTrainingConfig, EyeMaskDatasetConfig
from ..training.losses import BCEDiceCriterion
from ..training.zarr_eye_mask_dataset import YOLOTargetStore, load_yolo_targets
from ..utils.system import get_git_info, get_environment_info
from .unet import UNetSmall

try:
    from torch.utils.tensorboard import SummaryWriter

    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False
    SummaryWriter = None  # type: ignore

from torch import amp


class EMA:
    def __init__(self, beta: float = 0.98):
        self.beta = float(beta)
        self.avg: Optional[float] = None

    def update(self, x: float) -> float:
        if self.avg is None:
            self.avg = float(x)
        else:
            self.avg = self.beta * self.avg + (1 - self.beta) * float(x)
        return self.avg


def _fmt_rate(samples_per_sec: float, bs: int) -> str:
    rate = samples_per_sec if math.isfinite(samples_per_sec) else 0.0
    return f"{rate:6.1f} samp/s (bs={bs})"


def _channel_to_rgb(channel: np.ndarray) -> np.ndarray:
    arr = np.clip(channel, 0.0, 1.0)
    return np.repeat(arr[None, ...], 3, axis=0)


def _compose_preview_grid(
    images: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    label_mode: str,
    max_samples: int = 4,
) -> Optional[np.ndarray]:
    if images.size == 0:
        return None
    samples = min(max_samples, images.shape[0])
    rows: List[np.ndarray] = []
    channels = [0, 1] if label_mode == "lr" else [0]
    for idx in range(samples):
        img = images[idx]
        if img.shape[0] == 1:
            img_rgb = np.repeat(img, 3, axis=0)
        elif img.shape[0] >= 3:
            img_rgb = img[:3]
        else:
            img_rgb = np.tile(img, (3 // img.shape[0] + 1, 1, 1))[:3]
        img_rgb = np.clip(img_rgb, 0.0, 1.0)
        parts: List[np.ndarray] = [np.transpose(img_rgb, (1, 2, 0))]
        for ch in channels:
            gt = targets[idx][ch] if targets[idx].ndim == 3 else targets[idx]
            pred = preds[idx][ch] if preds[idx].ndim == 3 else preds[idx]
            gt_rgb = np.transpose(_channel_to_rgb(gt), (1, 2, 0))
            pred_rgb = np.transpose(_channel_to_rgb(pred), (1, 2, 0))
            parts.extend([gt_rgb, pred_rgb])
        row = np.concatenate(parts, axis=1)
        rows.append(row)
    if not rows:
        return None
    grid = np.concatenate(rows, axis=0)
    return np.clip(grid, 0.0, 1.0)


class _StoreChunkCache:
    """Lightweight loader that batches reads per Zarr chunk."""

    def __init__(self, store: YOLOTargetStore) -> None:
        self.store = store
        total = int(store.roi_array.shape[0])
        chunk = getattr(store.roi_array, "chunks", (total,))[0] if hasattr(store.roi_array, "chunks") else total
        if not isinstance(chunk, int) or chunk <= 0:
            chunk = total
        self.chunk_size = max(1, chunk)
        self._current_chunk_id: Optional[int] = None
        self._roi_chunk: Optional[np.ndarray] = None
        self._target_chunk: Optional[np.ndarray] = None

    def _load_chunk(self, chunk_id: int) -> None:
        start = chunk_id * self.chunk_size
        stop = min(start + self.chunk_size, int(self.store.roi_array.shape[0]))

        roi_np = np.asarray(self.store.roi_array[start:stop])
        if roi_np.ndim == 3:
            roi_np = roi_np[:, None, :, :]
        if roi_np.ndim != 4:
            raise ValueError(f"Unexpected ROI chunk shape: {roi_np.shape}")
        roi_np = roi_np.astype(np.float32, copy=False)
        max_val = float(np.nanmax(roi_np)) if roi_np.size else 0.0
        if max_val > 1.0:
            roi_np /= 255.0
        roi_np = np.nan_to_num(roi_np, nan=0.0, posinf=1.0, neginf=0.0)
        roi_np = np.clip(roi_np, 0.0, 1.0)

        if self.store.use_probs:
            target_np = np.asarray(self.store.mask_probs_array[start:stop], dtype=np.float32)
        else:
            target_np = np.asarray(self.store.masks_array[start:stop], dtype=np.float32)
            max_val = float(np.nanmax(target_np)) if target_np.size else 0.0
            if max_val > 1.0:
                target_np /= 255.0
        if target_np.ndim == 3:
            target_np = target_np[:, None, :, :]
        if target_np.ndim != 4:
            raise ValueError(f"Unexpected mask chunk shape: {target_np.shape}")
        target_np = np.nan_to_num(target_np, nan=0.0, posinf=1.0, neginf=0.0)
        target_np = np.clip(target_np, 0.0, 1.0)

        self._current_chunk_id = chunk_id
        self._roi_chunk = roi_np
        self._target_chunk = target_np

    def get_sample(self, index: int, label_mode: str) -> Tuple[np.ndarray, np.ndarray]:
        chunk_id = index // self.chunk_size
        if self._current_chunk_id != chunk_id or self._roi_chunk is None or self._target_chunk is None:
            self._load_chunk(chunk_id)
        offset = index - chunk_id * self.chunk_size
        roi_np = self._roi_chunk[offset]
        target_np = self._target_chunk[offset]
        if label_mode == "union":
            if target_np.shape[0] == 1:
                target = target_np
            else:
                target = np.max(target_np, axis=0, keepdims=True)
        else:
            if target_np.shape[0] == 1:
                target = np.repeat(target_np, 2, axis=0)
            else:
                target = target_np
        return roi_np.copy(), target.copy()


class YOLOChunkedDataset(Dataset):
    """Dataset that groups samples by Zarr chunk to minimise redundant reads."""

    def __init__(
        self,
        entries: List[Tuple[YOLOTargetStore, np.ndarray]],
        label_mode: str,
        shuffle_chunks: bool,
        seed: int,
    ) -> None:
        self.label_mode = label_mode
        self.records: List[Tuple[int, int]] = []
        self.groups: List[List[int]] = []
        self.caches: List[_StoreChunkCache] = []
        self.stores: List[YOLOTargetStore] = []

        rng = np.random.default_rng(seed)
        for store_idx, (store, indices) in enumerate(entries):
            indices = np.asarray(indices, dtype=np.int64)
            if indices.size == 0:
                continue
            chunk_size = getattr(store.roi_array, "chunks", (int(store.roi_array.shape[0]),))[0]
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                chunk_size = int(store.roi_array.shape[0])
            chunk_map: Dict[int, List[int]] = {}
            for idx in indices:
                chunk_id = int(idx // chunk_size)
                chunk_map.setdefault(chunk_id, []).append(int(idx))
            chunk_ids = list(chunk_map.keys())
            if shuffle_chunks:
                rng.shuffle(chunk_ids)
            else:
                chunk_ids.sort()

            self.caches.append(_StoreChunkCache(store))
            self.stores.append(store)

            for chunk_id in chunk_ids:
                chunk_indices = sorted(chunk_map[chunk_id])
                group_positions: List[int] = []
                for idx_val in chunk_indices:
                    self.records.append((len(self.stores) - 1, idx_val))
                    group_positions.append(len(self.records) - 1)
                self.groups.append(group_positions)

        if not self.records:
            raise ValueError("No samples available to build dataset.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        store_idx, roi_idx = self.records[index]
        roi_np, target_np = self.caches[store_idx].get_sample(roi_idx, self.label_mode)
        return {
            "img": roi_np,
            "masks": target_np,
        }


class ChunkSampler(Sampler[int]):
    """Sampler that iterates samples chunk-by-chunk with optional reshuffling each epoch."""

    def __init__(self, dataset: YOLOChunkedDataset, seed: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        order = list(range(len(self.dataset.groups)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)
        for group_idx in order:
            group = self.dataset.groups[group_idx]
            for item_idx in group:
                yield item_idx
        self.epoch += 1

    def __len__(self) -> int:
        return len(self.dataset)


def _worker_init_fn(_: int) -> None:
    os.environ.setdefault("BLOSC_NTHREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)


def _collate_pairs(batch: Sequence[Dict[str, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.from_numpy(np.stack([sample["img"] for sample in batch], axis=0)).float()
    masks = torch.from_numpy(np.stack([sample["masks"] for sample in batch], axis=0)).float()
    return imgs, masks


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device_str)
    if device_str.lower() == "cpu":
        return torch.device("cpu")
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    return torch.device(device_str)


def _dataset_split_indices(
    total: int,
    split_train: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if total <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    indices = np.arange(total, dtype=np.int64)
    rng.shuffle(indices)
    if total == 1:
        return indices, np.empty((0,), dtype=np.int64)

    train_count = int(round(split_train * total))
    train_count = max(1, min(total - 1, train_count))

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]
    return train_indices, val_indices


def _load_arrays_for_dataset(
    name: str,
    cfg: EyeMaskDatasetConfig,
    label_mode: str,
    run_override: Optional[str],
    eye_masks_method: Optional[str],
    mask_preference: str,
) -> YOLOTargetStore:
    try:
        store = load_yolo_targets(
            cfg.zarr_path,
            run_override or cfg.mask_run,
            label_mode,
            eye_masks_method,
            mask_preference,
        )
    except Exception as exc:  # pragma: no cover - safety net for runtime datasets
        raise RuntimeError(f"Failed to load dataset '{name}' from {cfg.zarr_path}: {exc}") from exc

    store.meta = dict(store.meta)
    store.meta["dataset_name"] = name
    store.meta["zarr_path"] = str(cfg.zarr_path)
    store.meta["mask_run_source"] = run_override or cfg.mask_run or store.meta.get("eye_masks_run")
    return store


def _assemble_training_datasets(
    config: EyeMaskTrainingConfig,
    console: Console,
) -> Tuple[YOLOChunkedDataset, YOLOChunkedDataset, List[Dict[str, object]]]:
    label_mode = config.training_params.label_mode
    run_override = config.training_params.eye_masks_run
    rng_seed = int(config.random_seed)

    train_entries: List[Tuple[YOLOTargetStore, np.ndarray]] = []
    val_entries: List[Tuple[YOLOTargetStore, np.ndarray]] = []
    meta_list: List[Dict[str, object]] = []

    for ds_idx, (name, ds_cfg) in enumerate(config.datasets.items()):
        store = _load_arrays_for_dataset(
            name,
            ds_cfg,
            label_mode,
            run_override,
            config.training_params.eye_masks_method,
            ds_cfg.mask_preference,
        )
        meta = dict(store.meta)
        total = int(meta.get("length", store.roi_array.shape[0]))

        if total == 0:
            console.print(f"[yellow]Warning:[/yellow] dataset '{name}' produced zero samples; skipping.")
            continue

        meta_list.append(meta)

        split_cfg = ds_cfg.split or config.default_split
        rng = np.random.default_rng(rng_seed + ds_idx)
        train_local, val_local = _dataset_split_indices(total, float(split_cfg.train), rng)

        if train_local.size == 0:
            raise ValueError(f"Training split for dataset '{name}' is empty; adjust split configuration.")
        if val_local.size == 0:
            val_local = train_local[:1].copy()

        train_entries.append((store, train_local))
        val_entries.append((store, val_local))

    if not train_entries:
        raise ValueError("No datasets produced training samples; check configuration.")

    train_dataset = YOLOChunkedDataset(train_entries, label_mode, shuffle_chunks=True, seed=rng_seed)
    val_dataset = YOLOChunkedDataset(val_entries, label_mode, shuffle_chunks=False, seed=rng_seed + 10_000)

    return train_dataset, val_dataset, meta_list


def _summarise_datasets(
    meta_list: Iterable[Dict[str, object]],
    console: Console,
) -> None:
    table = Table(title="Loaded Eye-Mask Datasets", show_header=True)
    table.add_column("Dataset", justify="left", style="cyan")
    table.add_column("Eye Mask Run", style="magenta")
    table.add_column("Method", style="magenta")
    table.add_column("Crop Run", style="magenta")
    table.add_column("Samples", justify="right", style="green")
    table.add_column("Source", justify="left", style="yellow")
    table.add_column("Probabilities", justify="center", style="blue")
    table.add_column("Eye Labels", justify="left", style="white")

    for meta in meta_list:
        table.add_row(
            str(meta.get("dataset_name", "unknown")),
            str(meta.get("eye_masks_run", "unknown")),
            str(meta.get("eye_masks_method", "unknown")),
            str(meta.get("crop_run", "unknown")),
            f"{int(meta.get('length', 0)):,}",
            str(meta.get("zarr_path", "")),
            "mask_probs" if bool(meta.get("use_probs", False)) else "binarized",
            ", ".join(meta.get("eye_labels", [])) if meta.get("eye_labels") else "--",
        )
    console.print(table)
    console.print()


def _compute_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> Tuple[float, List[float]]:
    probs = logits.sigmoid()
    if probs.ndim == 3:
        probs = probs.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    intersection = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    dice_mean = float(dice.mean().item())
    per_channel = [float(val.item()) for val in dice.mean(dim=0)]
    return dice_mean, per_channel


def _create_run_directory(base_dir: Path, run_name: Optional[str]) -> Path:
    base_dir = base_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_dir = base_dir / run_name
        if run_dir.exists():
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            run_dir = base_dir / f"{run_name}_{timestamp}"
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = base_dir / f"unet_{timestamp}"
        suffix = 1
        while run_dir.exists():
            run_dir = base_dir / f"unet_{timestamp}_{suffix:03d}"
            suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_checkpoint(
    path: Path,
    model: UNetSmall,
    history: List[Dict[str, float]],
    config: EyeMaskTrainingConfig,
    meta_list: List[Dict[str, object]],
    best_dice: float,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": asdict(model.config),
        "label_mode": config.training_params.label_mode,
        "label_source": config.training_params.label_source,
        "config": config.model_dump(mode="json"),
        "dataset_meta": meta_list,
        "training_history": history,
        "best_val_dice": float(best_dice),
    }
    torch.save(checkpoint, path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a U-Net eye mask segmenter using YOLO segmentation outputs stored in Zarr."
    )
    parser.add_argument("config_path", help="Path to EyeMaskTrainingConfig YAML.")
    parser.add_argument(
        "--run-name",
        help="Optional name for the training run directory (defaults to timestamp).",
    )
    parser.add_argument(
        "--output-dir",
        help="Override training_params.project for saving checkpoints and logs.",
    )
    parser.add_argument(
        "--device",
        help="Override training_params.device (e.g. 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Optional epoch override (defaults to config value).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable rich progress bars; print epoch summaries only.",
    )
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default=None,
        help="TensorBoard log directory (enables TB logging if provided and TensorBoard is installed).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile even if available.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    console.print("[bold cyan]Starting U-Net eye-mask training[/bold cyan]\n")

    cfg_path = Path(args.config_path)
    config = EyeMaskTrainingConfig.from_yaml(cfg_path)

    train_dataset, val_dataset, meta_list = _assemble_training_datasets(config, console)
    _summarise_datasets(meta_list, console)

    train_params = config.training_params
    device = _resolve_device(args.device or train_params.device)
    epochs = int(args.epochs or train_params.epochs)
    batch_size = int(train_params.batch)
    num_workers = int(config.num_workers)
    rng_seed = int(config.random_seed)

    default_run_name = f"unet_{(config.training_params.label_mode or 'union').lower()}"
    run_dir = _create_run_directory(
        Path(args.output_dir or train_params.project or "runs/eye_masks"),
        args.run_name or default_run_name,
    )
    console.print(f"[green]Output directory:[/green] {run_dir}\n")

    writer = None
    if args.tb_logdir and TB_AVAILABLE:
        tb_dir = Path(args.tb_logdir).expanduser().resolve()
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        console.print(f"[cyan]TensorBoard:[/cyan] logging to {tb_dir}")
    elif args.tb_logdir and not TB_AVAILABLE:
        console.print("[yellow]TensorBoard not available; install tensorboard to enable logging.[/yellow]")

    preview_dir = run_dir / "validation_previews"

    torch.manual_seed(int(config.random_seed))
    np.random.seed(int(config.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.random_seed))

    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch._dynamo.config.suppress_errors = True

    train_sampler = ChunkSampler(train_dataset, seed=rng_seed, shuffle=True)
    val_sampler = ChunkSampler(val_dataset, seed=rng_seed + 1, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=bool(num_workers > 0),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=_collate_pairs,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    val_worker_count = max(0, num_workers // 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=val_worker_count,
        prefetch_factor=2 if val_worker_count > 0 else None,
        persistent_workers=bool(val_worker_count > 0),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=_collate_pairs,
        worker_init_fn=_worker_init_fn if val_worker_count > 0 else None,
    )

    sample = train_dataset[0]
    in_channels = int(sample["img"].shape[0])
    out_channels = 1 if config.training_params.label_mode == "union" else 2
    model = UNetSmall(in_channels=in_channels, out_channels=out_channels, base_channels=32).to(device)
    compile_enabled_flag = False
    if args.no_compile:
        console.print("[yellow]torch.compile disabled (--no-compile).[/yellow]")
    elif importlib.util.find_spec("networkx") is None:
        console.print("[yellow]torch.compile skipped (networkx not available).[/yellow]")
    else:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            compile_enabled_flag = True
        except Exception as exc:
            console.print(
                f"[yellow]torch.compile failed ({exc.__class__.__name__}); running eager mode.[/yellow]"
            )
    if compile_enabled_flag:
        console.print("[cyan]torch.compile:[/cyan] enabled")

    criterion = BCEDiceCriterion(
        bce_weight=float(train_params.bce_weight) if train_params.bce_weight is not None else 0.5
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_params.lr0),
        weight_decay=float(train_params.weight_decay),
    )
    use_autocast = device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_autocast)

    history: List[Dict[str, float]] = []
    best_dice = float("-inf")
    best_epoch = -1

    git_info = get_git_info()
    env_info = get_environment_info()

    console.print(f"[cyan]Training device:[/cyan] {device}")
    console.print(f"[cyan]Epochs:[/cyan] {epochs}, [cyan]Batch size:[/cyan] {batch_size}\n")

    use_progress = not args.no_progress

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_samples = 0

        ema_loss = EMA()
        ema_rate = EMA(beta=0.90)

        train_progress = None
        if use_progress:
            train_progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold]Train[/bold]"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("{task.fields[loss]}"),
                TextColumn("•"),
                TextColumn("{task.fields[rate]}"),
                TimeElapsedColumn(),
                TextColumn("• ETA"),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            )
            train_progress.__enter__()
            train_task = train_progress.add_task(
                f"[blue]Epoch {epoch+1}/{epochs}[/blue]",
                total=len(train_loader),
                loss="loss=----",
                rate="-----",
            )

        _t0 = time.time()
        for step, (imgs, masks) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_eff = imgs.size(0)
            train_loss_total += float(loss.item()) * batch_size_eff
            train_samples += batch_size_eff

            batch_time = max(1e-6, time.time() - _t0)
            samples_per_sec = batch_size_eff / batch_time
            sm_loss = ema_loss.update(float(loss.item()))
            sm_rate = ema_rate.update(samples_per_sec)

            if use_progress and train_progress is not None:
                train_progress.update(
                    train_task,
                    advance=1,
                    loss=f"loss={sm_loss:.4f}",
                    rate=_fmt_rate(sm_rate, batch_size_eff),
                )
            _t0 = time.time()

        if train_progress is not None:
            train_progress.__exit__(None, None, None)

        train_loss = train_loss_total / max(1, train_samples)

        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        dice_accum: List[float] = []
        dice_per_channel_accum: List[List[float]] = []
        preview_written = False

        val_progress = None
        if use_progress:
            val_progress = Progress(
                SpinnerColumn(style="magenta"),
                TextColumn("[bold]Val  [/bold]"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("{task.fields[loss]}"),
                TextColumn("• Dice {task.fields[dice]}"),
                console=console,
                transient=False,
            )
            val_progress.__enter__()
            val_task = val_progress.add_task(
                f"[magenta]Epoch {epoch+1}/{epochs}[/magenta]",
                total=len(val_loader),
                loss="loss=----",
                dice="----",
            )

        with torch.no_grad():
            for step, (imgs, masks) in enumerate(val_loader, 1):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with amp.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                batch_size_eff = imgs.size(0)
                val_loss_total += float(loss.item()) * batch_size_eff
                val_samples += batch_size_eff

                dice_mean, dice_channels = _compute_dice(logits, masks)
                dice_accum.append(dice_mean)
                dice_per_channel_accum.append(dice_channels)

                if not preview_written:
                    imgs_cpu = imgs.detach().cpu().float().numpy()
                    targets_cpu = masks.detach().cpu().float().numpy()
                    preds_cpu = torch.sigmoid(logits.detach()).cpu().float().numpy()
                    preview_grid = _compose_preview_grid(
                        imgs_cpu,
                        targets_cpu,
                        preds_cpu,
                        config.training_params.label_mode,
                    )
                    if preview_grid is not None:
                        preview_dir.mkdir(parents=True, exist_ok=True)
                        preview_path = preview_dir / f"epoch_{epoch + 1:03d}.png"
                        Image.fromarray((preview_grid * 255).astype(np.uint8)).save(preview_path)
                        if writer is not None:
                            writer.add_image(
                                "validation/preview",
                                torch.from_numpy(preview_grid.transpose(2, 0, 1)).float(),
                                epoch + 1,
                            )
                        preview_written = True
                        console.print(
                            f"[cyan]Saved validation preview:[/cyan] {preview_path}"
                        )

                if use_progress and val_progress is not None:
                    val_progress.update(
                        val_task,
                        advance=1,
                        loss=f"loss={float(loss.item()):.4f}",
                        dice=f"{float(dice_mean):.4f}",
                    )

        if val_progress is not None:
            val_progress.__exit__(None, None, None)

        val_loss = val_loss_total / max(1, val_samples)
        val_dice = float(np.mean(dice_accum)) if dice_accum else float("nan")
        per_channel_mean: List[float] = []
        if dice_per_channel_accum:
            per_channel_mean = list(np.mean(dice_per_channel_accum, axis=0))

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_dice": float(val_dice),
            }
        )

        summary = Table(title=f"Epoch {epoch+1}/{epochs} Summary", show_header=True)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="yellow")
        summary.add_row("train_loss", f"{train_loss:.4f}")
        summary.add_row("val_loss", f"{val_loss:.4f}")
        summary.add_row("val_dice", f"{val_dice:.4f}")
        if per_channel_mean:
            for idx, val in enumerate(per_channel_mean):
                summary.add_row(f"val_dice_ch{idx}", f"{val:.4f}")
        console.print(summary)

        if writer is not None:
            global_step = epoch + 1
            writer.add_scalar("loss/train", train_loss, global_step)
            writer.add_scalar("loss/val", val_loss, global_step)
            writer.add_scalar("dice/val_mean", val_dice, global_step)
            if per_channel_mean:
                for idx, val in enumerate(per_channel_mean):
                    writer.add_scalar(f"dice/val_ch{idx}", float(val), global_step)

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            _save_checkpoint(run_dir / "best_model.pt", model, history, config, meta_list, best_dice)

    _save_checkpoint(run_dir / "last_model.pt", model, history, config, meta_list, best_dice)

    summary = {
        "run_dir": str(run_dir),
        "best_val_dice": float(best_dice),
        "best_epoch": int(best_epoch),
        "epochs": int(epochs),
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "label_mode": config.training_params.label_mode,
        "label_source": config.training_params.label_source,
        "device": str(device),
        "git_commit": git_info.get("commit_hash", "unknown"),
        "git_branch": git_info.get("branch", "unknown"),
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(run_dir / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(run_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    with open(run_dir / "dataset_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta_list, fh, indent=2)

    console.print(
        f"\n[green]Training complete.[/green] Best val dice={best_dice:.4f} (epoch {best_epoch}). "
        f"Checkpoints saved in {run_dir}"
    )

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
