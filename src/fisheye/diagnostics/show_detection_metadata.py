#!/usr/bin/env python3
"""
Display the metadata emitted by YOLO inference (detect_yolo) for quick inspection.

This script reads both root-level attributes and the selected detect_run attrs so
you can confirm what information downstream workers will see (source/inference
resolutions, palette provenance, codec, fps, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import zarr
from rich.console import Console
from rich.table import Table


def _collect_attrs(group: zarr.Group, keys: Iterable[str]) -> Dict[str, Optional[object]]:
    return {key: group.attrs.get(key) for key in keys}


def _format_resolution(value: Optional[object]) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]}×{value[1]}"
    return str(value)


def _add_kv_rows(table: Table, rows: List[Tuple[str, Optional[object]]]) -> None:
    for label, value in rows:
        if isinstance(value, float):
            display = f"{value:.3f}"
        elif isinstance(value, (list, tuple)):
            display = ", ".join(str(v) for v in value)
        elif value is None:
            display = "—"
        else:
            display = str(value)
        table.add_row(label, display)


def show_detection_metadata(zarr_path: Path, detect_run: Optional[str]) -> None:
    console = Console()
    console.print(f"[bold]Opening detection archive:[/bold] {zarr_path}")
    root = zarr.open(str(zarr_path), mode="r")

    if "detect_runs" not in root:
        raise ValueError(f"Archive '{zarr_path}' has no detect_runs group.")
    detect_runs = root["detect_runs"]

    if detect_run is None:
        detect_run = detect_runs.attrs.get("latest")
        if not detect_run:
            raise ValueError("detect_runs group missing 'latest' attribute; specify --detect-run.")
        console.print(f"[cyan]Using detect_run:[/cyan] {detect_run} (latest)")
    else:
        if detect_run not in detect_runs:
            raise ValueError(f"detect_run '{detect_run}' not found under detect_runs/.")
        console.print(f"[cyan]Using detect_run:[/cyan] {detect_run}")
    detect_group = detect_runs[detect_run]

    # Root-level metadata
    root_keys = [
        "source_video_width",
        "source_video_height",
        "source_full_width",
        "source_full_height",
        "source_video_resolution",
        "source_full_resolution",
        "inference_resolution",
        "resized_for_inference",
        "fps",
        "n_frames",
        "duration_seconds",
        "video_codec",
        "video_pix_fmt",
        "palette_video_width",
        "palette_video_height",
        "palette_original_resolution",
        "palette_downsampled_resolution",
        "palette_downsample_method",
        "palette_fps",
        "palette_total_frames",
        "palette_duration_seconds",
        "source_zarr_path",
    ]
    root_meta = _collect_attrs(root, root_keys)

    table_root = Table(title="Root Attributes", show_lines=True, box=None)
    table_root.add_column("Field", style="bold")
    table_root.add_column("Value")
    _add_kv_rows(table_root, [(key, root_meta[key]) for key in root_keys if key in root_meta])

    console.print(table_root)

    # raw_video metadata if present (common when sharing archive with import)
    if "raw_video" in root:
        raw = root["raw_video"]
        raw_keys = [
            "fps",
            "total_frames",
            "original_resolution",
            "video_width",
            "video_height",
            "video_codec",
            "video_pix_fmt",
            "video_duration_seconds",
            "has_full_resolution",
            "has_downsampled",
            "downsampled_resolution",
            "downsample_method",
        ]
        raw_meta = _collect_attrs(raw, raw_keys)
        table_raw = Table(title="raw_video Attributes", show_lines=True, box=None)
        table_raw.add_column("Field", style="bold")
        table_raw.add_column("Value")
        _add_kv_rows(table_raw, [(key, raw_meta[key]) for key in raw_keys if key in raw_meta])
        console.print(table_raw)

    # Detect-run metadata
    detect_keys = [
        "source_video_width",
        "source_video_height",
        "source_full_width",
        "source_full_height",
        "inference_width",
        "inference_height",
        "palette_original_resolution",
        "palette_downsampled_resolution",
        "palette_has_full_resolution",
        "palette_has_downsampled",
        "source_zarr_path",
        "detection_source",
        "model_name",
        "model_path",
    ]
    detect_meta = _collect_attrs(detect_group, detect_keys)

    table_detect = Table(title=f"detect_runs/{detect_run} Attributes", show_lines=True, box=None)
    table_detect.add_column("Field", style="bold")
    table_detect.add_column("Value")
    _add_kv_rows(table_detect, [(key, detect_meta[key]) for key in detect_keys if key in detect_meta])

    console.print(table_detect)

    console.print("[green]Metadata inspection complete.[/green]")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show detection metadata generated by detect_yolo.")
    parser.add_argument("zarr_path", type=Path, help="Path to detection Zarr archive.")
    parser.add_argument("--detect-run", help="Specific detect run name (defaults to latest).")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    show_detection_metadata(args.zarr_path, args.detect_run)


if __name__ == "__main__":  # pragma: no cover
    main()
