#!/usr/bin/env python3
"""Verify detection → crop → keypoints → eye-mask provenance consistency."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table


@dataclass
class ProvenanceInfo:
    detect_run: Optional[str]
    crop_run: Optional[str]
    keypoint_run: Optional[str]
    eye_mask_run: Optional[str]


def _latest(group: Optional[zarr.Group]) -> Optional[str]:
    if group is None:
        return None
    latest = group.attrs.get("latest")
    if isinstance(latest, bytes):
        latest = latest.decode("utf-8", "ignore")
    if isinstance(latest, str) and latest in group:
        return latest
    return None


def _resolve_detection_array(root: zarr.Group, path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    normalized = path.strip("/")
    if normalized not in root:
        return None
    group = root[normalized]
    arr = group.get("detection_source")
    return arr[:].astype(np.int8, copy=False) if arr is not None else None


def _load_detection_provenance(root: zarr.Group) -> tuple[Optional[str], Optional[np.ndarray]]:
    detect_parent = root.get("detect_runs")
    run = _latest(detect_parent)
    if run is None:
        return None, None
    group = detect_parent[run]
    arr = group.get("detection_source")
    return run, arr[:].astype(np.int8, copy=False) if arr is not None else None


def _load_crop_provenance(root: zarr.Group) -> tuple[Optional[str], Optional[str], Optional[str], Optional[np.ndarray]]:
    parent = root.get("crop_runs")
    run = _latest(parent)
    if run is None:
        return None, None, None, None
    group = parent[run]
    detect_run = group.attrs.get("source_detect_run")
    detect_path = group.attrs.get("detection_source_path")
    arr = group.get("detection_source")
    return run, detect_run, detect_path, arr[:].astype(np.int8, copy=False) if arr is not None else None


def _load_keypoint_provenance(root: zarr.Group) -> tuple[Optional[str], Optional[str], Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    parent = root.get("keypoints_runs")
    run = _latest(parent)
    if run is None:
        return None, None, None, None, None
    group = parent[run]
    detect_run = group.attrs.get("source_detect_run")
    crop_run = group.attrs.get("source_crop_run")
    det_arr = group.get("detection_source")
    heading_valid = group.get("heading_valid")
    return run, detect_run, crop_run, det_arr[:] if det_arr is not None else None, heading_valid[:] if heading_valid is not None else None


def _load_eye_mask_provenance(root: zarr.Group) -> tuple[Optional[str], Optional[str], Optional[str], Optional[np.ndarray]]:
    parent = root.get("eye_masks_runs")
    run = _latest(parent)
    if run is None:
        return None, None, None, None
    group = parent[run]
    detect_run = group.attrs.get("source_detect_run")
    keypoint_run = group.attrs.get("source_keypoint_run")
    arr = group.get("detection_source")
    return run, detect_run, keypoint_run, arr[:] if arr is not None else None


def _compare_arrays(console: Console, name: str, lhs: Optional[np.ndarray], rhs: Optional[np.ndarray]) -> bool:
    if lhs is None or rhs is None:
        console.print(f"[yellow]{name}: missing data; cannot compare[/yellow]")
        return False
    if lhs.shape != rhs.shape:
        console.print(f"[red]{name}: length mismatch {lhs.shape[0]} vs {rhs.shape[0]}[/red]")
        return False
    mismatch = np.sum(lhs != rhs)
    if mismatch:
        console.print(f"[red]{name}: {mismatch} differing entries[/red]")
        return False
    console.print(f"[green]{name}: OK ({lhs.shape[0]} entries)")
    return True


def check_provenance(zarr_path: str, console: Optional[Console] = None) -> int:
    console = console or Console()
    root = zarr.open(zarr_path, mode="r")

    detect_run, detect_arr = _load_detection_provenance(root)
    crop_run, crop_detect_attr, crop_detect_path, crop_arr = _load_crop_provenance(root)
    kp_run, kp_detect_attr, kp_crop_attr, kp_arr, heading_valid = _load_keypoint_provenance(root)
    eye_run, eye_detect_attr, eye_kp_attr, eye_arr = _load_eye_mask_provenance(root)

    summary = Table(title="Provenance Overview")
    summary.add_column("Stage")
    summary.add_column("Run")
    summary.add_column("Source detect")
    summary.add_column("Source crop")
    summary.add_column("Source keypoints")

    summary.add_row("detect", detect_run or "–", "–", "–", "–")
    summary.add_row("crop", crop_run or "–", crop_detect_attr or "–", crop_detect_path or "–", "–")
    summary.add_row("keypoints", kp_run or "–", kp_detect_attr or "–", kp_crop_attr or "–", "–")
    summary.add_row("eye_masks", eye_run or "–", eye_detect_attr or "–", "–", eye_kp_attr or "–")
    console.print(summary)

    status = 0

    if detect_run and crop_run:
        if crop_detect_attr != detect_run:
            console.print(f"[red]Crop run points to detect '{crop_detect_attr}', latest is '{detect_run}'[/red]")
            status = 1
        else:
            console.print("[green]Crop run references latest detect run.[/green]")
    if kp_run:
        if kp_detect_attr and detect_run and kp_detect_attr != detect_run:
            console.print(f"[red]Keypoints detect source '{kp_detect_attr}' != latest '{detect_run}'[/red]")
            status = 1
        if kp_crop_attr and crop_run and kp_crop_attr != crop_run:
            console.print(f"[red]Keypoints crop source '{kp_crop_attr}' != latest '{crop_run}'[/red]")
            status = 1
    if eye_run:
        if eye_detect_attr and detect_run and eye_detect_attr != detect_run:
            console.print(f"[red]Eye masks detect source '{eye_detect_attr}' != latest '{detect_run}'[/red]")
            status = 1
        if eye_kp_attr and kp_run and eye_kp_attr != kp_run:
            console.print(f"[red]Eye masks keypoint source '{eye_kp_attr}' != latest '{kp_run}'[/red]")
            status = 1

    console.print("\n[bold]Detection source array comparisons[/bold]")
    if crop_arr is not None:
        reference_arr = detect_arr
        if crop_detect_path:
            resolved = _resolve_detection_array(root, crop_detect_path)
            if resolved is not None:
                reference_arr = resolved
            else:
                console.print(f"[yellow]Detection source path '{crop_detect_path}' missing detection_source array; falling back to latest detect run.[/yellow]")
        status |= 0 if _compare_arrays(console, "source vs crop", reference_arr, crop_arr) else 1
    else:
        console.print("[yellow]Crop run missing detection_source array.[/yellow]")
        status = 1

    if kp_arr is not None:
        status |= 0 if _compare_arrays(console, "crop vs keypoints", crop_arr, kp_arr) else 1
    else:
        console.print("[yellow]Keypoint run missing detection_source array.[/yellow]")
        status = 1

    if eye_arr is not None:
        status |= 0 if _compare_arrays(console, "keypoints vs eye_masks", kp_arr, eye_arr) else 1
    else:
        console.print("[yellow]Eye mask run missing detection_source array.[/yellow]")
        status = 1

    if heading_valid is not None and kp_arr is not None:
        invalid = np.sum(np.logical_and(heading_valid, kp_arr == 1))
        if invalid:
            console.print(f"[red]Heading valid includes {invalid} synthetic detections[/red]")
            status = 1
        else:
            console.print("[green]Heading valid excludes synthetic detections.[/green]")
    else:
        console.print("[yellow]Heading valid array unavailable; skipping heading check.[/yellow]")

    return status


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to Palette Zarr archive")
    args = parser.parse_args(argv)
    console = Console()
    try:
        return check_provenance(args.zarr_path, console)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
