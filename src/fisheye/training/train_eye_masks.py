"""YOLO segmentation trainer for eye-mask generation from Zarr datasets."""

from __future__ import annotations

import argparse
import platform
import time
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.segment import SegmentationTrainer

from .config import EyeMaskTrainingConfig
from .zarr_eye_mask_dataset import build_eye_mask_datasets, EyeMaskDatasetBundle
from ..utils.system import get_git_info


class YoloSegDataLoader(DataLoader):
    """Minimal wrapper that matches Ultralytics expectations."""

    def reset(self) -> None:  # pragma: no cover - behaviour defined by YOLO internals
        pass


def seg_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    import numpy as np

    images = torch.from_numpy(np.stack([sample["img"] for sample in batch])).float()
    im_files = [sample["im_file"] for sample in batch]
    ori_shapes = [sample["ori_shape"] for sample in batch]
    ratio_pads = [sample["ratio_pad"] for sample in batch]

    cls_list: List[torch.Tensor] = []
    bbox_list: List[torch.Tensor] = []
    batch_idx_list: List[torch.Tensor] = []
    mask_tensors: List[torch.Tensor] = []
    segment_tensors: List[torch.Tensor] = []

    for i, sample in enumerate(batch):
        cls_arr = sample["cls"]
        if isinstance(cls_arr, np.ndarray) and cls_arr.size > 0:
            cls_tensor = torch.from_numpy(cls_arr).reshape(-1, 1)
            bbox_tensor = torch.from_numpy(sample["bboxes"])
            mask_np = sample["masks"]
            masks_tensor = torch.from_numpy(mask_np).float()

            cls_list.append(cls_tensor)
            bbox_list.append(bbox_tensor)
            batch_idx_list.append(torch.full((cls_tensor.shape[0],), i, dtype=torch.long))
            mask_tensors.append(masks_tensor)

            segments_np = sample["segments"]
            if isinstance(segments_np, list):
                for seg in segments_np:
                    segment_tensors.append(torch.from_numpy(seg).float())
        else:
            # Maintain ordering by appending empty tensors to preserve instance alignment when batching
            mask_shape = sample["masks"].shape if isinstance(sample["masks"], np.ndarray) else (0, images.shape[2], images.shape[3])
            mask_tensors.append(torch.zeros(mask_shape, dtype=torch.float32))

    if cls_list:
        cls = torch.cat(cls_list, 0).float()
        bboxes = torch.cat(bbox_list, 0).float()
        batch_idx = torch.cat(batch_idx_list, 0)
        masks = torch.cat([m for m in mask_tensors if m.numel() > 0], 0) if any(m.numel() > 0 for m in mask_tensors) else torch.zeros((0, images.shape[2], images.shape[3]), dtype=torch.float32)
    else:
        cls = torch.empty((0, 1), dtype=torch.float32)
        bboxes = torch.empty((0, 4), dtype=torch.float32)
        batch_idx = torch.empty((0,), dtype=torch.long)
        masks = torch.zeros((0, images.shape[2], images.shape[3]), dtype=torch.float32)

    return {
        "img": images,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "masks": masks,
        "segments": segment_tensors,
        "im_file": im_files,
        "ori_shape": ori_shapes,
        "ratio_pad": ratio_pads,
    }


class ZarrSegmentationTrainer(SegmentationTrainer):
    """Custom trainer that reads segmentation batches from Zarr datasets."""

    dataset_bundle: EyeMaskDatasetBundle
    batch_size_override: int = 16
    num_workers_override: int = 8

    def plot_training_labels(self):  # noqa: D401 - disable Ultralytics plotting step that expects cached labels
        return None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):  # noqa: D401 - signature defined upstream
        dataset = self.dataset_bundle.train_dataset if mode == "train" else self.dataset_bundle.val_dataset
        shuffle = mode == "train"
        loader_kwargs = {
            "batch_size": self.batch_size_override,
            "shuffle": shuffle,
            "num_workers": self.num_workers_override,
            "collate_fn": seg_collate_fn,
            "pin_memory": True,
            "persistent_workers": False,
        }

        return YoloSegDataLoader(
            dataset,
            **loader_kwargs,
        )


def _display_dataset_stats(bundle: EyeMaskDatasetBundle, console: Console) -> None:
    for stats in bundle.stats.values():
        table = Table(title=f"📦 {stats.dataset_name}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Zarr Path", stats.zarr_path)
        table.add_row("Crop Run", stats.crop_run)
        table.add_row("Mask Run", stats.mask_run)
        table.add_row("Total ROIs", f"{stats.total_rois:,}")
        table.add_row("Positive ROIs", f"{stats.positive_rois:,}")
        table.add_row("Negative ROIs", f"{stats.negative_rois:,}")
        if stats.background_negatives:
            table.add_row("Background Negatives", f"{stats.background_negatives:,}")
        console.print(table)
        console.print()


def _build_training_report(
    config: EyeMaskTrainingConfig,
    bundle: EyeMaskDatasetBundle,
    results,
    training_start_time: float,
    training_duration_seconds: float,
    run_name: str,
) -> Dict[str, object]:
    git_info = get_git_info()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(training_start_time))
    save_dir = Path(results.save_dir)
    results_csv = save_dir / "results.csv"

    final_metrics: Dict[str, float] = {}
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        for key in [
            "metrics/seg_P",
            "metrics/seg_R",
            "metrics/seg_mAP50",
            "metrics/seg_mAP50-95",
            "train/box_loss",
            "train/seg_loss",
            "train/cls_loss",
        ]:
            if key in last:
                final_metrics[key] = float(last[key])

    dataset_summaries = {
        name: {
            "zarr_path": stats.zarr_path,
            "crop_run": stats.crop_run,
            "mask_run": stats.mask_run,
            "total_rois": stats.total_rois,
            "positive_rois": stats.positive_rois,
            "negative_rois": stats.negative_rois,
            "background_negatives": stats.background_negatives,
        }
        for name, stats in bundle.stats.items()
    }

    report = config.model_dump()
    report["dataset_summary"] = dataset_summaries
    report["training_history"] = {
        "training_run_name": run_name,
        "output_directory": str(save_dir),
        "final_model_path": str(save_dir / "weights" / "best.pt"),
        "training_start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_start_time)),
        "training_duration_hours": round(training_duration_seconds / 3600.0, 3),
        "git_commit_hash": git_info.get("commit_hash", "N/A"),
        "git_branch": git_info.get("branch", "N/A"),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "ultralytics_version": ultralytics_version,
        "cuda_available": torch.cuda.is_available(),
        "final_metrics": final_metrics,
    }
    report["timestamp"] = timestamp
    return report


def main(args: argparse.Namespace) -> None:
    console = Console()
    console.print("[bold cyan]Starting YOLO Eye-Mask Segmentation Training[/bold cyan]\n")

    try:
        config = EyeMaskTrainingConfig.from_yaml(Path(args.config_path))
        console.print(f"[bold green]✓ Loaded config:[/bold green] {args.config_path}\n")
    except Exception as exc:
        console.print(f"[bold red]✗ Failed to load config:[/bold red] {exc}")
        return

    try:
        console.print("[bold cyan]Preparing datasets...[/bold cyan]")
        bundle = build_eye_mask_datasets(config, console=console)
    except Exception as exc:
        console.print(f"[bold red]✗ Failed to prepare datasets:[/bold red] {exc}")
        traceback.print_exc()
        return

    console.print("[bold cyan]Dataset Overview[/bold cyan]\n")
    _display_dataset_stats(bundle, console)

    model = YOLO(config.training_params.model)

    ZarrSegmentationTrainer.dataset_bundle = bundle
    ZarrSegmentationTrainer.batch_size_override = config.training_params.batch
    ZarrSegmentationTrainer.num_workers_override = config.num_workers

    training_start = time.time()
    console.print("[bold green]⚡ Starting Training...[/bold green]\n")

    train_kwargs = {}
    if config.training_params.max_det is not None:
        train_kwargs["max_det"] = config.training_params.max_det
    if config.training_params.conf is not None:
        train_kwargs["conf"] = config.training_params.conf

    try:
        results = model.train(
            trainer=ZarrSegmentationTrainer,
            data=args.config_path,
            name=args.run_name or "eye_mask_segmentation",
            epochs=config.training_params.epochs,
            batch=config.training_params.batch,
            imgsz=config.training_params.imgsz,
            lr0=config.training_params.lr0,
            momentum=config.training_params.momentum,
            weight_decay=config.training_params.weight_decay,
            patience=config.training_params.patience,
            device=config.training_params.device,
            project=config.training_params.project,
            overlap_mask=False,
            **train_kwargs,
        )
    except Exception as exc:
        console.print(f"[bold red]✗ Training failed:[/bold red] {exc}")
        traceback.print_exc()
        return

    training_duration = time.time() - training_start
    run_name = results.save_dir.name if hasattr(results, "save_dir") else (args.run_name or "eye_mask_segmentation")

    console.print("\n[bold cyan]Generating Training Report...[/bold cyan]")
    report: Dict[str, object] | None = None
    try:
        report = _build_training_report(
            config=config,
            bundle=bundle,
            results=results,
            training_start_time=training_start,
            training_duration_seconds=training_duration,
            run_name=run_name,
        )
        timestamp = report["timestamp"]
        save_dir = Path(results.save_dir)
        report_path = save_dir / f"{timestamp}_eye_mask_training_report.yaml"
        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)
        console.print(f"[bold green]✓ Training report saved:[/bold green] {report_path}")
    except Exception as exc:
        console.print(f"[bold red]✗ Failed to save training report:[/bold red] {exc}")
        traceback.print_exc()

    if report and "training_history" in report and "final_metrics" in report["training_history"]:
        metrics = report["training_history"]["final_metrics"]
        if metrics:
            table = Table(title=" Final Training Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            for key, value in metrics.items():
                table.add_row(key, f"{value:.4f}")
            console.print(table)

    console.print("\n[bold green]✓ Training Complete![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO segmentation model for eye masks using Zarr datasets")
    parser.add_argument("config_path", type=str, help="Path to eye mask training config YAML")
    parser.add_argument("--run-name", type=str, help="Optional name for the training run directory")
    parsed_args = parser.parse_args()
    main(parsed_args)
