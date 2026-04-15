"""Shared console rendering helpers for training entrypoints."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def print_training_banner(console: Console, task_label: str) -> None:
    title = f"Zarr YOLO {task_label} Training - Fisheye Module"
    width = max(55, len(title) + 6)
    line = "═" * width
    console.print(f"[bold cyan]{line}[/bold cyan]")
    console.print(f"[bold cyan]{title:^{width}}[/bold cyan]")
    console.print(f"[bold cyan]{line}[/bold cyan]\n")


def print_section_header(console: Console, title: str, *, style: str = "bold cyan") -> None:
    console.print(f"[{style}]{title}[/{style}]")
    console.rule()


def print_training_hyperparameters(
    console: Console,
    training_params: Mapping[str, Any],
    *,
    loader_overrides: Optional[Mapping[str, Any]] = None,
    include_loader_note: bool = False,
) -> None:
    if loader_overrides is not None:
        console.print(
            Panel(
                json.dumps(dict(loader_overrides), indent=2),
                title="[bold cyan]Custom Loader Augmentations[/bold cyan]",
                expand=False,
            )
        )
        if include_loader_note:
            console.print(
                "[dim]Ultralytics built-in mosaic/mixup/cutmix/copy_paste/auto_augment are held neutral for Zarr loader runs.[/dim]"
            )
        console.print()

    console.print(
        Panel(
            json.dumps(dict(training_params), indent=2),
            title="[bold yellow]Training Hyperparameters[/bold yellow]",
            expand=False,
        )
    )
    console.print()


def print_training_start(console: Console, *, lightning: bool = False) -> None:
    label = "⚡ Starting Training..." if lightning else "Starting Training..."
    console.print(f"[bold green]{label}[/bold green]\n")


def _format_source_type(source_type: Any) -> str:
    text = str(source_type or "unknown")
    normalized = text.strip().lower()
    if normalized == "detect":
        return "[cyan]detect[/cyan] (original)"
    if normalized == "filtered":
        return "[yellow]filtered[/yellow] (jumps removed)"
    if normalized == "interpolated":
        return "[magenta]interpolated[/magenta] (gaps filled)"
    if normalized == "manual":
        return "[green]manual[/green] (manual review)"
    return text


def print_dataset_details(
    console: Console,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    task: str,
    pose_schema: Optional[Mapping[str, Any]] = None,
) -> None:
    task_norm = str(task or "").strip().lower()
    if task_norm == "pose" and pose_schema:
        labels = pose_schema.get("keypoint_labels") or []
        skeleton = pose_schema.get("skeleton") or []
        kpt_shape = pose_schema.get("kpt_shape")
        schema_lines = [
            f"kpt_shape: {kpt_shape}" if kpt_shape else "kpt_shape: unknown",
            f"keypoints: {', '.join(str(v) for v in labels)}" if labels else "keypoints: unknown",
            f"skeleton: {skeleton}" if skeleton else "skeleton: none",
        ]
        console.print(
            Panel(
                "\n".join(schema_lines),
                title="[bold cyan]Pose Schema[/bold cyan]",
                expand=False,
            )
        )

    printed_tables = 0
    for dataset_name, meta in metadata.items():
        if not isinstance(meta, Mapping):
            console.print(f"[red]✗ {dataset_name}: invalid metadata payload[/red]")
            continue
        if "error" in meta:
            console.print(f"[red]✗ {dataset_name}: {meta['error']}[/red]")
            continue

        table = Table(title=f"📦 {dataset_name}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Video Frames", str(meta.get("video_frames", "N/A")))
        table.add_row("FPS", str(meta.get("fps", "N/A")))

        crop_info = meta.get("crop_info") if isinstance(meta.get("crop_info"), Mapping) else {}
        if crop_info:
            table.add_row("Crop Source", _format_source_type(crop_info.get("source_type")))
            total_rois = crop_info.get("total_rois", crop_info.get("n_rois", 0))
            table.add_row("Total ROIs", f"{int(total_rois or 0):,}")
            roi_size = crop_info.get("roi_size")
            if roi_size:
                table.add_row("ROI Size", str(roi_size))
            if bool(crop_info.get("includes_interpolated", False)):
                n_real = int(crop_info.get("n_real", crop_info.get("n_real_detections", 0)) or 0)
                n_interp = int(
                    crop_info.get("n_interpolated", crop_info.get("n_interpolated_detections", 0)) or 0
                )
                table.add_row("  └─ Real ROIs", f"{n_real:,}")
                table.add_row("  └─ Interpolated ROIs", f"{n_interp:,}")

        if task_norm == "detect":
            det_info = meta.get("detection_info") if isinstance(meta.get("detection_info"), Mapping) else {}
            if det_info:
                table.add_row("Total Detections", f"{int(det_info.get('total_detections', 0) or 0):,}")
                table.add_row("Detection Rate", f"{float(det_info.get('detection_rate', 0.0) or 0.0):.1f}%")
            quality = meta.get("data_quality") if isinstance(meta.get("data_quality"), Mapping) else {}
            if bool(quality.get("has_refinement", False)):
                if "jumps_removed" in quality:
                    table.add_row("Jumps Removed", str(quality.get("jumps_removed")))
                if "gaps_filled" in quality:
                    table.add_row("Gaps Filled", str(quality.get("gaps_filled")))
                manual_present = quality.get("manual_edited_detections")
                if manual_present is not None:
                    table.add_row("Manual-Edited Detections", f"{int(manual_present or 0):,}")
        elif task_norm == "pose":
            tracking = meta.get("tracking_info") if isinstance(meta.get("tracking_info"), Mapping) else {}
            if "warning" in tracking:
                table.add_row("Keypoints", str(tracking["warning"]))
            else:
                table.add_row("Keypoint Run", str(tracking.get("run_name", "N/A")))
                refined_run = tracking.get("refined_run")
                if refined_run:
                    table.add_row("Refined Run", str(refined_run))
                table.add_row("Keypoints Processed", str(int(tracking.get("keypoints_processed", 0) or 0)))
                usable = tracking.get("usable_keypoints")
                total = tracking.get("total_keypoints")
                usable_rate = tracking.get("usable_keypoints_rate")
                if usable_rate is not None:
                    if usable is not None and total is not None:
                        table.add_row("Usable Keypoints", f"{int(usable):,}/{int(total):,}")
                    table.add_row("Usable Keypoint Rate", f"{float(usable_rate):.3f}")
                else:
                    table.add_row("Raw Keypoint Success", f"{float(tracking.get('success_rate', 0.0) or 0.0):.2f}")
                labels = tracking.get("keypoint_labels")
                if labels:
                    table.add_row("Keypoint Labels", ", ".join(str(v) for v in labels))

        console.print(table)
        console.print()
        printed_tables += 1

    if printed_tables == 0 and metadata:
        console.print("[yellow]No valid dataset metadata tables to display.[/yellow]")
