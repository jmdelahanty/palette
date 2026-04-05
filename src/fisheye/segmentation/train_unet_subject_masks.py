"""Train a lightweight U-Net to predict unified subject-mask probabilities."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch._dynamo
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import amp
from torch.utils.data import DataLoader, Sampler

from ..training.config import SubjectMaskTrainingConfig
from ..training.losses import MaskedBCEDiceCriterion
from ..training.training_run_shared import (
    record_registry_training_run as _shared_record_registry_training_run,
)
from ..training.zarr_subject_mask_dataset import (
    SubjectMaskChunkedDataset,
    build_subject_mask_training_datasets,
)
from ..utils.system import build_invocation_record, get_environment_info, get_git_info
from .unet import UNetSmall


class EMA:
    def __init__(self, beta: float = 0.98) -> None:
        self.beta = float(beta)
        self.avg: Optional[float] = None

    def update(self, x: float) -> float:
        if self.avg is None:
            self.avg = float(x)
        else:
            self.avg = self.beta * self.avg + (1 - self.beta) * float(x)
        return self.avg


class ChunkSampler(Sampler[int]):
    """Sampler that iterates samples chunk-by-chunk with optional reshuffling."""

    def __init__(self, dataset: SubjectMaskChunkedDataset, seed: int, shuffle: bool = True) -> None:
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
            for item_idx in self.dataset.groups[group_idx]:
                yield item_idx
        self.epoch += 1

    def __len__(self) -> int:
        return len(self.dataset)


def _fmt_rate(samples_per_sec: float, bs: int) -> str:
    rate = samples_per_sec if math.isfinite(samples_per_sec) else 0.0
    return f"{rate:6.1f} samp/s (bs={bs})"


def _worker_init_fn(_: int) -> None:
    os.environ.setdefault("BLOSC_NTHREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)


def _collate_triplets(batch: Sequence[Dict[str, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs = torch.from_numpy(np.stack([sample["img"] for sample in batch], axis=0)).float()
    masks = torch.from_numpy(np.stack([sample["masks"] for sample in batch], axis=0)).float()
    valid = torch.from_numpy(np.stack([sample["valid_channels"] for sample in batch], axis=0)).float()
    return imgs, masks, valid


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device_str)
    if device_str.lower() == "cpu":
        return torch.device("cpu")
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    return torch.device(device_str)


def _compute_masked_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_channels: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[float, List[float]]:
    probs = logits.sigmoid()
    if probs.ndim == 3:
        probs = probs.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if valid_channels.ndim != 2:
        raise ValueError(f"Expected valid_channels shape (B,C), got {tuple(valid_channels.shape)}")

    valid = valid_channels.to(dtype=probs.dtype)
    valid_expanded = valid[:, :, None, None]
    intersection = (probs * targets * valid_expanded).sum(dim=(2, 3))
    denom = ((probs + targets) * valid_expanded).sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)

    valid_mask = valid_channels > 0
    if torch.any(valid_mask):
        dice_mean = float(dice[valid_mask].mean().item())
    else:
        dice_mean = float("nan")

    per_channel: List[float] = []
    for channel_idx in range(dice.shape[1]):
        channel_values = dice[:, channel_idx][valid_mask[:, channel_idx]]
        per_channel.append(float(channel_values.mean().item()) if channel_values.numel() else float("nan"))
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


def _sanitize_slug(value: Optional[str], fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or fallback


def _resolve_output_base_dir(
    *,
    output_dir: Optional[str],
    configured_project: Optional[str],
    set_id: Optional[str],
    config_path: Path,
    console: Console,
    nvme_root: Optional[Path] = None,
) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    configured_text = str(configured_project or "").strip()
    if configured_text:
        configured_path = Path(configured_text).expanduser()
        if configured_path.is_absolute():
            return configured_path.resolve()
        normalized = configured_text.replace("\\", "/").strip()
        normalized = normalized[2:] if normalized.startswith("./") else normalized
        if normalized.lower() not in {"runs/subject_masks"}:
            return configured_path.resolve()

    root = (nvme_root or Path("/nvme1")).expanduser()
    if root.exists():
        slug = _sanitize_slug(set_id, _sanitize_slug(config_path.stem, "subject_masks_training"))
        project_dir = (root / "models" / "subject_masks" / slug).resolve()
        console.print(f"[cyan]Using default model output directory:[/cyan] {project_dir}")
        return project_dir

    if configured_text:
        return Path(configured_text).expanduser().resolve()
    return Path("runs/subject_masks").resolve()


def _save_checkpoint(
    path: Path,
    model: UNetSmall,
    history: List[Dict[str, float]],
    config: SubjectMaskTrainingConfig,
    meta_list: List[Dict[str, object]],
    best_dice: float,
    mask_labels: Sequence[str],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": asdict(model.config),
        "label_schema_id": config.training_params.label_schema_id,
        "mask_labels": list(mask_labels),
        "config": config.model_dump(mode="json"),
        "dataset_meta": meta_list,
        "training_history": history,
        "best_val_dice": float(best_dice),
    }
    torch.save(checkpoint, path)


def _read_manifest_set_id(manifest_path: Optional[Path]) -> Optional[str]:
    if manifest_path is None:
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    raw_set_id = payload.get("set_id")
    if raw_set_id is None:
        return None
    set_id = str(raw_set_id).strip()
    return set_id or None


def _record_registry_training_run(
    *,
    args: argparse.Namespace,
    console: Console,
    invocation_payload: Optional[Dict[str, object]],
    run_id: str,
    set_id: Optional[str],
    config_path: Optional[Path],
    manifest_path: Optional[Path],
    model_path: Optional[Path],
    metrics_path: Optional[Path],
    status: str,
    final_metrics: Optional[Dict[str, object]],
) -> None:
    _shared_record_registry_training_run(
        args=args,
        console=console,
        invocation_payload=invocation_payload,
        run_id=run_id,
        set_id=set_id,
        config_path=config_path,
        manifest_path=manifest_path,
        model_path=model_path,
        metrics_path=metrics_path,
        status=status,
        final_metrics=final_metrics,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a U-Net subject-mask segmenter using merged training Zarr artifacts."
    )
    parser.add_argument("config_path", help="Path to SubjectMaskTrainingConfig YAML.")
    parser.add_argument("--manifest", type=str, help="Optional manifest JSON path to record in the registry.")
    parser.add_argument("--set-id", type=str, help="Optional training set ID.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument(
        "--log-registry",
        dest="log_registry",
        action="store_true",
        default=True,
        help="Record this training run in the registry (default: enabled).",
    )
    parser.add_argument(
        "--no-log-registry",
        dest="log_registry",
        action="store_false",
        help="Disable registry logging for this training run.",
    )
    parser.add_argument("--run-name", help="Optional name for the training run directory.")
    parser.add_argument("--output-dir", help="Override training_params.project for saving checkpoints and logs.")
    parser.add_argument("--device", help="Override training_params.device (e.g. 'cuda:0', 'cpu').")
    parser.add_argument("--epochs", type=int, help="Optional epoch override (defaults to config value).")
    parser.add_argument("--no-progress", action="store_true", help="Disable rich progress bars.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if available.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    cfg_path = Path(args.config_path)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    effective_set_id = str(args.set_id).strip() if args.set_id else None
    registry_run_id = args.run_name or f"unet_subject_{int(time.time())}"
    invocation_payload = (
        build_invocation_record(tool="fisheye.segmentation.train_unet_subject_masks", args=args)
        if args.log_registry
        else None
    )
    if not effective_set_id and manifest_path is not None:
        effective_set_id = _read_manifest_set_id(manifest_path)

    console.print("[bold cyan]Starting U-Net subject-mask training[/bold cyan]\n")

    try:
        config = SubjectMaskTrainingConfig.from_yaml(cfg_path)
    except Exception as exc:
        if args.log_registry:
            _record_registry_training_run(
                args=args,
                console=console,
                invocation_payload=invocation_payload,
                run_id=registry_run_id,
                set_id=effective_set_id,
                config_path=cfg_path,
                manifest_path=manifest_path,
                model_path=None,
                metrics_path=None,
                status="failed",
                final_metrics={
                    "stage": "config_load",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        raise

    try:
        bundle = build_subject_mask_training_datasets(config, console)
    except Exception as exc:
        if args.log_registry:
            _record_registry_training_run(
                args=args,
                console=console,
                invocation_payload=invocation_payload,
                run_id=registry_run_id,
                set_id=effective_set_id,
                config_path=cfg_path,
                manifest_path=manifest_path,
                model_path=None,
                metrics_path=None,
                status="failed",
                final_metrics={
                    "stage": "dataset_load",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        raise

    summary_table = Table(title="Loaded Subject-Mask Training Artifacts", show_header=True)
    summary_table.add_column("Dataset", style="cyan")
    summary_table.add_column("Subject Run", style="magenta")
    summary_table.add_column("Crop Run", style="magenta")
    summary_table.add_column("Train", justify="right", style="green")
    summary_table.add_column("Val", justify="right", style="green")
    summary_table.add_column("Schema", style="yellow")
    for meta in bundle.meta_list:
        summary_table.add_row(
            str(meta.get("dataset_name", "unknown")),
            str(meta.get("subject_mask_run", "unknown")),
            str(meta.get("crop_run", "unknown")),
            f"{int(meta.get('train_rows', 0)):,}",
            f"{int(meta.get('val_rows', 0)):,}",
            str(meta.get("label_schema_id", "unknown")),
        )
    console.print(summary_table)
    console.print()

    train_params = config.training_params
    device = _resolve_device(args.device or train_params.device)
    epochs = int(args.epochs or train_params.epochs)
    batch_size = int(train_params.batch_size)
    num_workers = int(config.num_workers)
    rng_seed = int(config.random_seed)

    default_run_name = f"unet_{bundle.label_schema_id}"
    project_base_dir = _resolve_output_base_dir(
        output_dir=args.output_dir,
        configured_project=train_params.project,
        set_id=effective_set_id,
        config_path=cfg_path,
        console=console,
    )
    train_params.project = str(project_base_dir)
    run_dir = _create_run_directory(project_base_dir, args.run_name or default_run_name)
    registry_run_id = run_dir.name
    console.print(f"[green]Output directory:[/green] {run_dir}\n")

    if args.log_registry:
        _record_registry_training_run(
            args=args,
            console=console,
            invocation_payload=invocation_payload,
            run_id=registry_run_id,
            set_id=effective_set_id,
            config_path=cfg_path,
            manifest_path=manifest_path,
            model_path=None,
            metrics_path=None,
            status="in_progress",
            final_metrics={"stage": "preflight_and_training", "status_detail": "training_started"},
        )

    best_dice = float("-inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []
    success_summary: Optional[Dict[str, object]] = None

    try:
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)

        torch.backends.cudnn.benchmark = device.type == "cuda"
        torch._dynamo.config.suppress_errors = True

        train_sampler = ChunkSampler(bundle.train_dataset, seed=rng_seed, shuffle=True)
        val_sampler = ChunkSampler(bundle.val_dataset, seed=rng_seed + 1, shuffle=False)
        train_loader = DataLoader(
            bundle.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=bool(num_workers > 0),
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=_collate_triplets,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )
        val_worker_count = max(0, num_workers // 2)
        val_loader = DataLoader(
            bundle.val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=val_worker_count,
            prefetch_factor=2 if val_worker_count > 0 else None,
            persistent_workers=bool(val_worker_count > 0),
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=_collate_triplets,
            worker_init_fn=_worker_init_fn if val_worker_count > 0 else None,
        )

        sample = bundle.train_dataset[0]
        in_channels = int(sample["img"].shape[0])
        out_channels = int(sample["masks"].shape[0])
        model = UNetSmall(in_channels=in_channels, out_channels=out_channels, base_channels=32).to(device)
        compile_enabled = False
        if args.no_compile:
            console.print("[yellow]torch.compile disabled (--no-compile).[/yellow]")
        elif importlib.util.find_spec("networkx") is None:
            console.print("[yellow]torch.compile skipped (networkx not available).[/yellow]")
        else:
            try:
                model = torch.compile(model)  # type: ignore[attr-defined]
                compile_enabled = True
            except Exception as exc:
                console.print(
                    f"[yellow]torch.compile failed ({exc.__class__.__name__}); running eager mode.[/yellow]"
                )
        if compile_enabled:
            console.print("[cyan]torch.compile:[/cyan] enabled")

        criterion = MaskedBCEDiceCriterion(
            bce_weight=float(train_params.bce_weight) if train_params.bce_weight is not None else 0.5,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_params.lr0),
            weight_decay=float(train_params.weight_decay),
        )
        use_autocast = device.type == "cuda"
        scaler = amp.GradScaler(enabled=use_autocast)

        console.print(f"[cyan]Training device:[/cyan] {device}")
        console.print(f"[cyan]Epochs:[/cyan] {epochs}, [cyan]Batch size:[/cyan] {batch_size}")
        console.print(f"[cyan]Label schema:[/cyan] {bundle.label_schema_id} -> {list(bundle.mask_labels)}\n")

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
                    f"[blue]Epoch {epoch + 1}/{epochs}[/blue]",
                    total=len(train_loader),
                    loss="loss=----",
                    rate="-----",
                )

            _t0 = time.time()
            for imgs, masks, valid_channels in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                valid_channels = valid_channels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
                    logits = model(imgs)
                    loss = criterion(logits, masks, valid_channels)
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
                    f"[magenta]Epoch {epoch + 1}/{epochs}[/magenta]",
                    total=len(val_loader),
                    loss="loss=----",
                    dice="----",
                )

            with torch.no_grad():
                for imgs, masks, valid_channels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    valid_channels = valid_channels.to(device, non_blocking=True)

                    with amp.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
                        logits = model(imgs)
                        loss = criterion(logits, masks, valid_channels)

                    batch_size_eff = imgs.size(0)
                    val_loss_total += float(loss.item()) * batch_size_eff
                    val_samples += batch_size_eff

                    dice_mean, dice_channels = _compute_masked_dice(logits, masks, valid_channels)
                    if math.isfinite(dice_mean):
                        dice_accum.append(dice_mean)
                    dice_per_channel_accum.append(dice_channels)

                    if use_progress and val_progress is not None:
                        val_progress.update(
                            val_task,
                            advance=1,
                            loss=f"loss={float(loss.item()):.4f}",
                            dice=f"{float(dice_mean):.4f}" if math.isfinite(dice_mean) else "nan",
                        )

            if val_progress is not None:
                val_progress.__exit__(None, None, None)

            val_loss = val_loss_total / max(1, val_samples)
            val_dice = float(np.mean(dice_accum)) if dice_accum else float("nan")
            per_channel_mean: List[float] = []
            channel_matrix = np.asarray(dice_per_channel_accum, dtype=np.float64)
            for channel_idx in range(channel_matrix.shape[1]):
                finite = channel_matrix[:, channel_idx][np.isfinite(channel_matrix[:, channel_idx])]
                per_channel_mean.append(float(finite.mean()) if finite.size else float("nan"))

            history_entry: Dict[str, float] = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_dice": float(val_dice),
            }
            for idx, label in enumerate(bundle.mask_labels):
                history_entry[f"val_dice_{label}"] = float(per_channel_mean[idx])
            history.append(history_entry)

            summary = Table(title=f"Epoch {epoch + 1}/{epochs} Summary", show_header=True)
            summary.add_column("Metric", style="cyan")
            summary.add_column("Value", style="yellow")
            summary.add_row("train_loss", f"{train_loss:.4f}")
            summary.add_row("val_loss", f"{val_loss:.4f}")
            summary.add_row("val_dice", f"{val_dice:.4f}")
            for idx, label in enumerate(bundle.mask_labels):
                summary.add_row(f"val_dice_{label}", f"{per_channel_mean[idx]:.4f}")
            console.print(summary)

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                _save_checkpoint(
                    run_dir / "best_model.pt",
                    model,
                    history,
                    config,
                    bundle.meta_list,
                    best_dice,
                    bundle.mask_labels,
                )

        _save_checkpoint(
            run_dir / "last_model.pt",
            model,
            history,
            config,
            bundle.meta_list,
            best_dice,
            bundle.mask_labels,
        )

        git_info = get_git_info()
        env_info = get_environment_info()
        summary_payload = {
            "run_dir": str(run_dir),
            "best_val_dice": float(best_dice),
            "best_epoch": int(best_epoch),
            "epochs": int(epochs),
            "train_samples": int(len(bundle.train_dataset)),
            "val_samples": int(len(bundle.val_dataset)),
            "label_schema_id": bundle.label_schema_id,
            "mask_labels": list(bundle.mask_labels),
            "device": str(device),
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(run_dir / "training_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2)
        with open(run_dir / "training_history.json", "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        with open(run_dir / "dataset_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(bundle.meta_list, fh, indent=2)
        success_summary = summary_payload
    except Exception as exc:
        if args.log_registry:
            failed_model_path = run_dir / "best_model.pt"
            failed_metrics_path = run_dir / "training_history.json"
            _record_registry_training_run(
                args=args,
                console=console,
                invocation_payload=invocation_payload,
                run_id=registry_run_id,
                set_id=effective_set_id,
                config_path=cfg_path,
                manifest_path=manifest_path,
                model_path=failed_model_path if failed_model_path.exists() else None,
                metrics_path=failed_metrics_path if failed_metrics_path.exists() else None,
                status="failed",
                final_metrics={
                    "stage": "training_loop",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        raise

    if args.log_registry:
        best_model_path = run_dir / "best_model.pt"
        history_path = run_dir / "training_history.json"
        _record_registry_training_run(
            args=args,
            console=console,
            invocation_payload=invocation_payload,
            run_id=registry_run_id,
            set_id=effective_set_id,
            config_path=cfg_path,
            manifest_path=manifest_path,
            model_path=best_model_path if best_model_path.exists() else None,
            metrics_path=history_path if history_path.exists() else None,
            status="success",
            final_metrics={
                "stage": "completed",
                "status_detail": "training_complete",
                "best_val_dice": success_summary.get("best_val_dice") if success_summary else None,
                "label_schema_id": bundle.label_schema_id,
            },
        )

    console.print(
        f"\n[green]✓[/green] Training complete. Best validation Dice = {best_dice:.4f} "
        f"at epoch {best_epoch}. Artifacts written to [cyan]{run_dir}[/cyan]"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
