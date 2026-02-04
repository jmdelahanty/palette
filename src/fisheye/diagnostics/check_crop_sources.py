#!/usr/bin/env python3
"""
Inspect crop runs and confirm their detection metadata matches the intended source.

For each `crop_runs/<run>` entry this diagnostic reports:
  * which detect/refined path the run references
  * whether that path exists and matches the latest refined detections
  * whether the stored `frame_indices` mirror the source detections
  * whether `detection_source` metadata (real vs interpolated) is present

This helps catch cases where a crop run was accidentally created from the
original detect run even though interpolated refined detections were available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    normalized = str(path).strip().strip("/")
    return normalized or None


def _resolve_group(root: zarr.Group, path: Optional[str]) -> Optional[zarr.Group]:
    norm = _normalize_path(path)
    if not norm:
        return None
    try:
        return root[norm]
    except KeyError:
        return None


def _latest(parent: Optional[zarr.Group]) -> Optional[str]:
    if parent is None:
        return None
    latest = parent.attrs.get("latest")
    if isinstance(latest, str) and latest in parent:
        return latest
    return None


def _expected_refined_path(root: zarr.Group) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (parent_name, interpolated_path) for the latest refined run, if present.
    """
    parent_name = None
    refined_parent = None
    if "refined_detect_runs" in root:
        parent_name = "refined_detect_runs"
        refined_parent = root[parent_name]
    elif "refined_runs" in root:
        parent_name = "refined_runs"
        refined_parent = root[parent_name]

    if refined_parent is None:
        return None, None

    latest = _latest(refined_parent)
    if latest is None:
        return parent_name, None

    manual_label = refined_parent[latest].attrs.get("manual_review_latest")
    if not manual_label and "manual" in refined_parent[latest]:
        manual_label = "manual"
    expected_path = f"{parent_name}/{latest}/{manual_label or 'interpolated'}"
    return parent_name, expected_path


def _arrays_match(a: zarr.Array, b: zarr.Array, chunk: int = 500_000) -> bool:
    if a.shape != b.shape:
        return False
    length = a.shape[0]
    if length == 0:
        return True
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        a_slice = a[start:end]
        b_slice = b[start:end]
        if not np.array_equal(a_slice, b_slice):
            return False
    return True


def analyze_crop_run(
    root: zarr.Group,
    run_name: str,
    expected_refined_path: Optional[str],
) -> Tuple[str, str, str, str, str, List[str]]:
    crop_group = root[f"crop_runs/{run_name}"]
    source_type = str(crop_group.attrs.get("detection_source_type", "unknown"))
    source_path = crop_group.attrs.get("detection_source_path")
    normalized_path = _normalize_path(source_path)
    issues: List[str] = []

    path_status = "missing"
    path_exists = False
    frame_match = "—"

    source_group = _resolve_group(root, source_path)
    if normalized_path is None:
        issues.append("missing detection_source_path attr")
    elif source_group is None:
        issues.append(f"path '{normalized_path}' not found in zarr")
    else:
        path_exists = True
        if expected_refined_path and normalized_path == expected_refined_path:
            path_status = "latest refined"
        elif normalized_path.startswith("detect_runs/"):
            path_status = "detect"
        elif source_type == "manual":
            path_status = "manual"
        else:
            path_status = "custom"

        # Compare frame indices when source data is available
        if "frame_indices" in crop_group and "frame_indices" in source_group:
            try:
                match = _arrays_match(crop_group["frame_indices"], source_group["frame_indices"])
                frame_match = "yes" if match else "no"
                if not match:
                    issues.append("frame_indices differ from source run")
            except Exception as exc:  # pragma: no cover - diagnostic guard rail
                frame_match = "error"
                issues.append(f"frame_indices comparison error: {exc}")
        else:
            frame_match = "missing"

    # Detection source stats
    det_stats = "absent"
    if "detection_source" in crop_group:
        det_arr = crop_group["detection_source"][:]
        n_real = int(np.sum(det_arr == 0))
        n_interp = int(np.sum(det_arr == 1))
        det_stats = f"real={n_real}, interp={n_interp}"
        if source_type == "interpolated" and n_interp == 0:
            issues.append("source_type=interpolated but detection_source has 0 synthetic rows")
    elif source_type in {"filtered", "interpolated"}:
        issues.append("expected detection_source array for refined data but none found")

    if normalized_path and expected_refined_path and normalized_path != expected_refined_path:
        issues.append(
            f"path mismatch (expected {expected_refined_path}, found {normalized_path})"
        )

    if not path_exists:
        path_status = "missing"

    issues_display = "none" if not issues else " ; ".join(issues)

    return (
        run_name,
        source_type,
        normalized_path or "—",
        path_status,
        frame_match,
        det_stats,
        issues_display,
    )


def list_crop_runs(root: zarr.Group) -> List[str]:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return []
    run_names = [
        name for name in getattr(crop_parent, "group_keys", lambda: [])()
        if isinstance(name, str) and isinstance(crop_parent.get(name), zarr.Group)
    ]
    return sorted(run_names)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check that crop runs reference the expected detection sources."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--crop-run",
        action="append",
        help="Crop run(s) to inspect (default: all runs). Can be specified multiple times.",
    )
    args = parser.parse_args(argv)

    console = Console()
    root = zarr.open(str(args.zarr_path), mode="r")

    parent_name, expected_refined_path = _expected_refined_path(root)
    if expected_refined_path:
        console.print(
            f"[dim]Latest refined detections: {expected_refined_path} ({parent_name})[/dim]"
        )
    else:
        console.print("[yellow]No refined detection runs found or no latest run recorded.[/yellow]")

    available_runs = list_crop_runs(root)
    if not available_runs:
        console.print("[red]No crop runs found in archive.[/red]")
        return

    selected_runs: Iterable[str]
    if args.crop_run:
        missing = [name for name in args.crop_run if name not in available_runs]
        if missing:
            console.print(f"[red]Unknown crop run(s): {', '.join(missing)}[/red]")
            console.print(f"Available runs: {', '.join(available_runs)}")
            return
        selected_runs = args.crop_run
    else:
        selected_runs = available_runs

    table = Table(
        "Crop Run",
        "Source Type",
        "Source Path",
        "Path Status",
        "Frame Indices",
        "Detection Source",
        "Issues",
        expand=True,
    )

    for run_name in selected_runs:
        row = analyze_crop_run(root, run_name, expected_refined_path)
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    main()
