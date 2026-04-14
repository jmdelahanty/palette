#!/usr/bin/env python3
"""
Refinement Pipeline Visualizer

Visualizes the results of the refinement pipeline:
- Original detections
- Canonical curated refined surface
- Legacy sparse subgroups when present (filtered/interpolated/manual)

Shows trajectories, coverage, curated-row provenance, source-decision summaries,
and supports listing runs or focusing on a specific frame window.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import zarr

from fisheye.shared.refined_detect_curation import (
    build_source_detection_decision_summary,
    extract_present_curated_rows,
    has_curated_refined_source_detections_projection,
    has_curated_refined_detect_surface,
    resolve_curated_refined_detect_run,
)
from fisheye.shared.refined_detect_resolution import resolve_active_curated_refined_run_name

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"

INTERPOLATED_BARCODE_CMAP = ListedColormap(["#8b0000", "#228b22", "#8a2be2"])
INTERPOLATED_BARCODE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], INTERPOLATED_BARCODE_CMAP.N)


def _clip_frame_range(frame_counts: np.ndarray,
                      frame_range: Tuple[int, int]) -> Tuple[int, int]:
    """Clip requested frame range to the available data length."""
    start, end = frame_range
    total_frames = len(frame_counts)
    start = max(0, min(start, total_frames))
    end = max(start, min(end, total_frames))
    return start, end


def load_refined_stage(group, stage_name: str, fps: float,
                       frame_range: Optional[Tuple[int, int]] = None,
                       total_frames: Optional[int] = None) -> Dict:
    """Load data from a refinement stage (filtered or interpolated)."""
    manual_edit_flags = None
    manual_edited_count = 0
    source_detection_summary = None
    if has_curated_refined_detect_surface(group):
        curated_rows = extract_present_curated_rows(group)
        bbox_coords = curated_rows['bbox_norm_coords']
        frame_indices = curated_rows['frame_indices']
        detection_source = curated_rows['detection_source']
        manual_edit_flags = curated_rows.get('manual_edit_flags')
        if manual_edit_flags is not None:
            manual_edit_flags = np.asarray(manual_edit_flags, dtype=bool)
            manual_edited_count = int(np.count_nonzero(manual_edit_flags))
        if has_curated_refined_source_detections_projection(group):
            source_detection_summary = build_source_detection_decision_summary(group)
    else:
        bbox_coords = group['bbox_norm_coords'][:]
        frame_indices = group['frame_indices'][:]
        detection_source = None
        if 'detection_source' in group:
            detection_source = group['detection_source'][:]
    if 'frame_counts' in group:
        frame_counts = group['frame_counts'][:]
    else:
        if total_frames is not None:
            frame_counts = np.bincount(
                frame_indices.astype(np.int64, copy=False),
                minlength=max(int(total_frames), 0),
            )
        else:
            frame_counts = np.bincount(frame_indices.astype(np.int64, copy=False))

    start_offset = 0
    if frame_range is not None:
        clipped_start, clipped_end = _clip_frame_range(frame_counts, frame_range)
        mask = (frame_indices >= clipped_start) & (frame_indices < clipped_end)
        bbox_coords = bbox_coords[mask]
        frame_indices = frame_indices[mask]
        if detection_source is not None:
            detection_source = detection_source[mask]
        if manual_edit_flags is not None:
            manual_edit_flags = manual_edit_flags[mask]
            manual_edited_count = int(np.count_nonzero(manual_edit_flags))
        frame_counts = frame_counts[clipped_start:clipped_end]
        start_offset = clipped_start

    centroids = bbox_coords[:, :2]
    safe_fps = fps if fps else 1.0
    time_seconds = (frame_indices - start_offset) / safe_fps

    total_frames = len(frame_counts)
    coverage = float(np.sum(frame_counts > 0) / total_frames) if total_frames else 0.0

    return {
        'bbox_coords': bbox_coords,
        'frame_counts': frame_counts,
        'frame_indices': frame_indices,
        'centroids': centroids,
        'frames': frame_indices,
        'time_seconds': time_seconds,
        'detection_source': detection_source,
        'manual_edit_flags': manual_edit_flags,
        'manual_edited_detections': manual_edited_count,
        'source_detection_summary': source_detection_summary,
        'coverage': coverage,
        'total_detections': len(bbox_coords),
        'stage': stage_name,
        'start_offset': start_offset,
        'total_frames': total_frames,
    }


def load_original_detections(detect_group, quality_group, fps: float,
                             frame_range: Optional[Tuple[int, int]] = None) -> Dict:
    """Load original detection data."""
    bbox_coords = detect_group['bbox_norm_coords'][:]
    frame_indices = detect_group['frame_indices'][:]
    if 'frame_counts' in detect_group:
        frame_counts = detect_group['frame_counts'][:]
    else:
        frame_counts = np.bincount(frame_indices.astype(np.int64, copy=False))

    detection_quality_labels = None
    if quality_group is not None and 'detection_quality_labels' in quality_group:
        detection_quality_labels = quality_group['detection_quality_labels'][:]
        if detection_quality_labels.size != frame_indices.size:
            detection_quality_labels = detection_quality_labels[:frame_indices.size]

    start_offset = 0
    if frame_range is not None:
        clipped_start, clipped_end = _clip_frame_range(frame_counts, frame_range)
        mask = (frame_indices >= clipped_start) & (frame_indices < clipped_end)
        bbox_coords = bbox_coords[mask]
        frame_indices = frame_indices[mask]
        if detection_quality_labels is not None:
            clipped_labels = detection_quality_labels[:mask.size]
            detection_quality_labels = clipped_labels[mask[:clipped_labels.size]]
            if detection_quality_labels.size != frame_indices.size:
                detection_quality_labels = None
        frame_counts = frame_counts[clipped_start:clipped_end]
        start_offset = clipped_start

    centroids = bbox_coords[:, :2]
    safe_fps = fps if fps else 1.0
    time_seconds = (frame_indices - start_offset) / safe_fps

    total_frames = len(frame_counts)
    coverage = float(np.sum(frame_counts > 0) / total_frames) if total_frames else 0.0

    return {
        'bbox_coords': bbox_coords,
        'frame_counts': frame_counts,
        'frame_indices': frame_indices,
        'detection_quality_labels': detection_quality_labels,
        'centroids': centroids,
        'frames': frame_indices,
        'time_seconds': time_seconds,
        'coverage': coverage,
        'total_detections': len(bbox_coords),
        'stage': 'original',
        'start_offset': start_offset,
        'total_frames': total_frames,
    }


def compute_stage_stats(dataset: Dict) -> Dict:
    """Compute coverage and detection statistics for a stage."""
    frame_counts = dataset['frame_counts']
    total_frames = dataset.get('total_frames', len(frame_counts))
    detection_mask = frame_counts > 0
    frames_with_detections = int(detection_mask.sum())
    coverage_percent = float(frames_with_detections / total_frames * 100) if total_frames else 0.0
    total_detections = int(len(dataset['frame_indices']))
    return {
        'total_frames': total_frames,
        'frames_with_detections': frames_with_detections,
        'coverage_percent': coverage_percent,
        'total_detections': total_detections,
        'manual_edited_detections': int(dataset.get('manual_edited_detections', 0) or 0),
        'source_detection_summary': dict(dataset.get('source_detection_summary') or {}),
    }


def _is_detection_stage_group(group) -> bool:
    """Return True if a group looks like a refined detection stage payload."""
    try:
        return ('bbox_norm_coords' in group) and ('frame_indices' in group)
    except Exception:
        return False


def _resolve_manual_group_name(refined_group) -> Optional[str]:
    """Resolve the active manual review subgroup name, if present."""
    manual_label = refined_group.attrs.get('manual_review_latest')
    if isinstance(manual_label, (bytes, bytearray)):
        manual_label = manual_label.decode('utf-8', errors='ignore')
    if isinstance(manual_label, str) and manual_label in refined_group:
        return manual_label
    if 'manual' in refined_group:
        return 'manual'
    return None


def _iter_refined_stage_names(refined_group) -> List[str]:
    """
    Return refined subgroup names to plot, in display order.

    Preferred order: canonical curated surface, active manual subgroup, then
    legacy filtered/interpolated subgroups, followed by other valid stage
    groups in lexical order.
    """
    ordered: List[str] = []
    if has_curated_refined_detect_surface(refined_group):
        ordered.append("refined")

    manual_name = _resolve_manual_group_name(refined_group)
    if manual_name and manual_name not in ordered and _is_detection_stage_group(refined_group[manual_name]):
        ordered.append(manual_name)

    for default_name in ('filtered', 'interpolated'):
        if default_name in refined_group and _is_detection_stage_group(refined_group[default_name]):
            ordered.append(default_name)

    extras = sorted(
        name
        for name in refined_group.keys()
        if not name.startswith('_')
        and name not in ordered
        and _is_detection_stage_group(refined_group[name])
    )
    ordered.extend(extras)
    return ordered


def _stage_display_name(stage_name: str, manual_group_name: Optional[str]) -> str:
    """Build a user-facing title for a stage key."""
    if stage_name == 'original':
        return 'Original Detections'
    if stage_name == 'curated':
        return 'Canonical Curated Refined Surface'
    if stage_name == 'refined':
        return 'Canonical Curated Refined Surface'
    if stage_name == 'filtered':
        return 'Legacy Filtered Subgroup'
    if stage_name == 'interpolated':
        return 'Legacy Interpolated Subgroup'
    if manual_group_name and stage_name == manual_group_name:
        if stage_name == 'manual':
            return 'Legacy Manual Subgroup'
        return f'Legacy Manual Subgroup ({stage_name})'
    return f'Refined ({stage_name})'


def _coverage_stage_display_name(stage_name: str) -> str:
    if stage_name == "original":
        return "Original"
    if stage_name == "refined":
        return "Canonical refined"
    if stage_name == "filtered":
        return "Legacy filtered"
    if stage_name == "interpolated":
        return "Legacy interpolated"
    return stage_name.title()


def _resolve_curated_stage(
    root,
    *,
    requested_run: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[object]]:
    curated_detect_run = resolve_active_curated_refined_run_name(
        root,
        run_name=requested_run,
    )
    if curated_detect_run is None:
        return None, None, None
    curated_detect_group, _ = resolve_curated_refined_detect_run(
        root,
        run_name=curated_detect_run,
    )
    return curated_detect_run, None, curated_detect_group


def summarize_refined_runs(zarr_path: str) -> int:
    """List available refined runs with parameters and coverage."""
    root = zarr.open(str(zarr_path), mode='r')
    refined_root = None
    if REFINED_DETECT_GROUP in root:
        refined_root = root[REFINED_DETECT_GROUP]
    elif LEGACY_REFINED_DETECT_GROUP in root:
        refined_root = root[LEGACY_REFINED_DETECT_GROUP]

    if refined_root is None:
        print("\nNo refined detection runs found.")
        print("Run: python -m fisheye.refinement.refine_detect <zarr_path>")
        return 1

    latest = refined_root.attrs.get('latest')
    run_names: List[str] = [name for name in refined_root.keys() if not name.startswith('_')]

    if not run_names:
        print("\nNo refined detection runs recorded yet.")
        return 1

    print("\n" + "=" * 70)
    print("AVAILABLE REFINED RUNS")
    print("=" * 70)

    for name in sorted(run_names):
        run_group = refined_root[name]
        latest_tag = " [LATEST]" if name == latest else ""
        params = dict(run_group.attrs.get('parameters', {}))
        coverage_attr = run_group.attrs.get('coverage_comparison')
        coverage = dict(coverage_attr) if isinstance(coverage_attr, dict) else {}

        print(f"\n{name}{latest_tag}")
        if params:
            filters = params.get('filters_applied', [])
            if params.get('interpolation_enabled') is False:
                print("  Canonical surface: sparse curated instances (no gap interpolation)")
            else:
                max_gap = params.get('max_gap', '?')
                method = params.get('interpolation_method', '?')
                print(f"  Max gap: {max_gap} frames | Method: {method}")
            if filters:
                print(f"  Filters: {filters}")
        if coverage:
            for stage in ('original', 'refined', 'filtered', 'interpolated'):
                stage_cov = coverage.get(stage, {})
                if isinstance(stage_cov, dict) and stage_cov:
                    cov_pct = stage_cov.get('coverage_percent')
                    frames = stage_cov.get('frames_with_detections')
                    dets = stage_cov.get('total_detections')
                    if cov_pct is not None:
                        stage_label = _coverage_stage_display_name(stage)
                        print(f"  {stage_label:<20}: {cov_pct:.2f}% "
                              f"({frames} frames, {dets} detections)")
        summary_stats = run_group.attrs.get('summary_statistics')
        if isinstance(summary_stats, dict) and summary_stats.get('rows_manual_edited') is not None:
            print(f"  Manual edited rows: {int(summary_stats['rows_manual_edited'])}")

    print()
    return 0


def visualize_refinement_pipeline(zarr_path: str,
                                  refined_run: Optional[str] = None,
                                  save_path: Optional[str] = None,
                                  frame_range: Optional[Tuple[int, int]] = None,
                                  show: bool = True,
                                  show_curated: bool = False,
                                  curated_detect_run: Optional[str] = None) -> None:
    """
    Visualize the refinement pipeline results.

    Args:
        zarr_path: Path to zarr file.
        refined_run: Specific refined run to visualize (default: latest).
        save_path: Optional path to save the figure.
        frame_range: Optional (start, end) tuple to limit visualization (end exclusive).
    """
    print(f"\n{'=' * 70}")
    print("REFINEMENT PIPELINE VISUALIZATION")
    print(f"{'=' * 70}")

    root = zarr.open(str(zarr_path), mode='r')
    fps = root.attrs.get('fps', 60.0)

    refined_root = None
    if REFINED_DETECT_GROUP in root:
        refined_root = root[REFINED_DETECT_GROUP]
    elif LEGACY_REFINED_DETECT_GROUP in root:
        refined_root = root[LEGACY_REFINED_DETECT_GROUP]

    if refined_root is None:
        print("\nError: No refined runs found!")
        print("Run: python -m fisheye.refinement.refine_detect <zarr_path>")
        return

    if refined_run is None:
        refined_run = refined_root.attrs.get('latest')
        if refined_run is None:
            print("\nError: No refined runs found!")
            return

    if refined_run not in refined_root:
        print(f"\nError: Refined run '{refined_run}' not found!")
        return

    refined_group = refined_root[refined_run]

    source_detect = refined_group.attrs.get('source_detect_run')
    source_quality = refined_group.attrs.get('source_quality_run')

    print(f"\nRefined run: {refined_run}")
    print(f"Source detect: {source_detect}")
    print(f"Source quality: {source_quality}")

    params = dict(refined_group.attrs.get('parameters', {}))
    if params:
        print(f"\nParameters:")
        if params.get('interpolation_enabled') is False:
            print("  Interpolation: disabled")
        else:
            print(f"  Max gap: {params.get('max_gap', '?')} frames")
            print(f"  Method: {params.get('interpolation_method', '?')}")
        print(f"  Filters: {params.get('filters_applied', [])}")

    detect_group = root[f'detect_runs/{source_detect}']

    quality_group = None
    quality_reports = detect_group.get('quality_reports')

    if source_quality and source_quality not in (None, '', 'N/A'):
        try:
            quality_group = detect_group[f'quality_reports/{source_quality}']
        except KeyError:
            print(f"Warning: quality run '{source_quality}' not found under detect run '{source_detect}'.")
    if quality_group is None and quality_reports is not None:
        fallback_name = quality_reports.attrs.get('latest')
        if fallback_name and fallback_name in quality_reports:
            quality_group = quality_reports[fallback_name]
            print(f"Using fallback quality run: {fallback_name}")
        else:
            candidates = [name for name in quality_reports.keys() if not name.startswith('_')]
            if candidates:
                fallback_name = sorted(candidates)[-1]
                quality_group = quality_reports[fallback_name]
                print(f"Using fallback quality run: {fallback_name}")

    if quality_group is None:
        print("Warning: No quality report available; trajectory overlays will not show quality flags.")

    manual_group_name = _resolve_manual_group_name(refined_group)
    refined_stage_names = _iter_refined_stage_names(refined_group)
    if not refined_stage_names:
        print("\nError: Refined run does not contain any stage groups with frame_indices/bbox_norm_coords.")
        return

    print(f"Refined stages: {', '.join(refined_stage_names)}")

    datasets: Dict[str, Dict] = {}
    datasets['original'] = load_original_detections(detect_group, quality_group, fps, frame_range)
    curated_detect_run_name: Optional[str] = None
    curated_crop_run_name: Optional[str] = None
    curated_detect_group = None
    if show_curated:
        try:
            curated_detect_run_name, curated_crop_run_name, curated_detect_group = _resolve_curated_stage(
                root,
                requested_run=curated_detect_run,
            )
        except Exception as exc:
            print(f"Warning: unable to resolve curated refined run: {exc}")
            curated_detect_group = None
        if curated_detect_group is not None:
            print(f"Curated refined run: {curated_detect_run_name}")
            datasets['curated'] = load_refined_stage(
                curated_detect_group,
                'curated',
                fps,
                frame_range,
                total_frames=datasets['original']['total_frames'],
            )
    for stage_name in refined_stage_names:
        datasets[stage_name] = load_refined_stage(
            refined_group if stage_name == 'refined' else refined_group[stage_name],
            stage_name,
            fps,
            frame_range,
            total_frames=datasets['original']['total_frames'],
        )

    stage_stats = {stage: compute_stage_stats(data) for stage, data in datasets.items()}
    stage_order = list(datasets.keys())
    dataset_items = [(stage, datasets[stage]) for stage in stage_order]

    n_cols = len(stage_order)
    fig_width = max(20.0, 5.5 * n_cols)
    fig = plt.figure(figsize=(fig_width, 14))
    gs = gridspec.GridSpec(4, n_cols, figure=fig, hspace=0.35, wspace=0.25,
                           height_ratios=[1.2, 0.5, 0.8, 0.8])

    stage_names = {
        stage_name: _stage_display_name(stage_name, manual_group_name)
        for stage_name in stage_order
    }

    stage_colors = {
        'original': '#1f77b4',
        'curated': '#8c564b',
        'refined': '#8c564b',
        'filtered': '#2ca02c',
        'interpolated': '#9467bd',
    }
    if manual_group_name:
        stage_colors[manual_group_name] = '#8c564b'
    fallback_palette = [
        '#17becf',
        '#ff7f0e',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#d62728',
    ]
    fallback_index = 0
    for stage_name in stage_order:
        if stage_name not in stage_colors:
            stage_colors[stage_name] = fallback_palette[fallback_index % len(fallback_palette)]
            fallback_index += 1

    # Row 1: Trajectory plots
    for idx, (stage, data) in enumerate(dataset_items):
        ax = fig.add_subplot(gs[0, idx])

        if len(data['centroids']) > 0:
            if stage == 'original':
                quality_labels = data.get('detection_quality_labels')
                if quality_labels is not None and quality_labels.size == len(data['centroids']):
                    clean_mask = quality_labels == 0
                    if np.any(clean_mask):
                        ax.scatter(data['centroids'][clean_mask, 0],
                                   data['centroids'][clean_mask, 1],
                                   c='green', s=2, alpha=0.6, label='Clean')
                    jump_mask = quality_labels == 3
                    if np.any(jump_mask):
                        ax.scatter(data['centroids'][jump_mask, 0],
                                   data['centroids'][jump_mask, 1],
                                   c='red', s=20, marker='x', alpha=0.8, label='Jumps')
                    blip_mask = quality_labels == 2
                    if np.any(blip_mask):
                        ax.scatter(data['centroids'][blip_mask, 0],
                                   data['centroids'][blip_mask, 1],
                                   c='orange', s=10, marker='s', alpha=0.6, label='Blips')
                    ax.legend(loc='upper right', fontsize=8)
                else:
                    scatter = ax.scatter(data['centroids'][:, 0], data['centroids'][:, 1],
                                         c=data['frames'] - data['start_offset'],
                                         cmap='viridis', s=2, alpha=0.6)
                    plt.colorbar(scatter, ax=ax, label='Frame', pad=0.01)
            elif stage != 'original' and data['detection_source'] is not None:
                real_mask = data['detection_source'] == 0
                interp_mask = data['detection_source'] == 1

                if np.any(real_mask):
                    ax.scatter(data['centroids'][real_mask, 0],
                               data['centroids'][real_mask, 1],
                               c=data['frames'][real_mask] - data['start_offset'],
                               cmap='viridis', s=2, alpha=0.6, label='Real')

                if np.any(interp_mask):
                    ax.scatter(data['centroids'][interp_mask, 0],
                               data['centroids'][interp_mask, 1],
                               c='red', s=15, marker='o', alpha=0.7,
                               edgecolors='darkred', linewidths=1, label='Interpolated')
                ax.legend(loc='upper right', fontsize=8)
            else:
                scatter = ax.scatter(data['centroids'][:, 0], data['centroids'][:, 1],
                                     c=data['frames'] - data['start_offset'],
                                     cmap='viridis', s=2, alpha=0.6)
                plt.colorbar(scatter, ax=ax, label='Frame', pad=0.01)

        stats = stage_stats[stage]
        total_frames_stage = max(stats['total_frames'], 1)
        title = f"{stage_names[stage]}\n"
        title += f"Coverage: {stats['frames_with_detections']}/{total_frames_stage} " \
                 f"frames ({stats['coverage_percent']:.2f}%)"

        if stage == 'filtered':
            removed = stage_stats['original']['total_detections'] - stats['total_detections']
            coverage_loss = stage_stats['original']['coverage_percent'] - stats['coverage_percent']
            title += f"\nRemoved: {removed} ({coverage_loss:.2f}%)"
        elif stage == 'interpolated':
            baseline_stage = 'filtered' if 'filtered' in stage_stats else 'original'
            baseline_label = 'Filtered' if baseline_stage == 'filtered' else 'Original'
            added = stats['total_detections'] - stage_stats[baseline_stage]['total_detections']
            coverage_gain = stats['coverage_percent'] - stage_stats[baseline_stage]['coverage_percent']
            title += f"\nAdded vs {baseline_label}: {added} ({coverage_gain:.2f}%)"
        elif idx > 0:
            prev_stage = stage_order[idx - 1]
            delta = stats['total_detections'] - stage_stats[prev_stage]['total_detections']
            cov_delta = stats['coverage_percent'] - stage_stats[prev_stage]['coverage_percent']
            title += f"\nΔ vs prev: {delta:+d} ({cov_delta:+.2f}%)"
        if stats['manual_edited_detections'] > 0:
            title += f"\nManual edited: {stats['manual_edited_detections']}"
        source_detection_summary = stats.get('source_detection_summary') or {}
        total_candidates = int(source_detection_summary.get('total_candidates', 0) or 0)
        if total_candidates > 0:
            title += f"\nSource candidates: {total_candidates}"

        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('X Position (normalized)')
        ax.set_ylabel('Y Position (normalized)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.invert_yaxis()

    # Row 2: Detection presence barcode
    for idx, (stage, data) in enumerate(dataset_items):
        ax = fig.add_subplot(gs[1, idx])

        frame_counts = data['frame_counts']
        if frame_counts.size == 0:
            ax.text(0.5, 0.5, 'No frames', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        detection_mask = frame_counts > 0
        time_seconds = np.arange(len(detection_mask)) / max(fps, 1e-6)

        extent_end = time_seconds[-1] if time_seconds.size else 0
        imshow_kwargs = dict(
            aspect='auto',
            interpolation='nearest',
            extent=[0, extent_end, 0, 1],
        )

        if stage != 'original' and data.get('detection_source') is not None:
            status = np.zeros_like(frame_counts, dtype=np.int8)
            status[detection_mask] = 1

            rel_frames = (data['frame_indices'] - data['start_offset']).astype(np.int64, copy=False)
            valid = (rel_frames >= 0) & (rel_frames < status.size)
            if np.any(valid):
                det_source = data['detection_source']
                interp_idx = rel_frames[valid & (det_source == 1)]
                if interp_idx.size:
                    status[interp_idx] = 2
                real_idx = rel_frames[valid & (det_source == 0)]
                if real_idx.size:
                    status[real_idx] = np.maximum(status[real_idx], 1)

            barcode_data = status.reshape(1, -1)
            ax.imshow(
                barcode_data,
                cmap=INTERPOLATED_BARCODE_CMAP,
                norm=INTERPOLATED_BARCODE_NORM,
                **imshow_kwargs,
            )
            legend_handles = [
                Patch(facecolor=INTERPOLATED_BARCODE_CMAP.colors[0], label='No detection'),
                Patch(facecolor=INTERPOLATED_BARCODE_CMAP.colors[1], label='Real detection'),
                Patch(facecolor=INTERPOLATED_BARCODE_CMAP.colors[2], label='Interpolated'),
            ]
            ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.9)
        else:
            barcode_data = detection_mask.reshape(1, -1).astype(float)
            ax.imshow(
                barcode_data,
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                **imshow_kwargs,
            )

        gaps = []
        in_gap = False
        gap_start = None
        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_start = i
                in_gap = True
            elif has_det and in_gap:
                gap_size = i - gap_start
                if gap_size > 30:
                    gaps.append((gap_start / fps, i / fps, gap_size))
                in_gap = False
        if in_gap:
            gap_size = len(detection_mask) - gap_start
            if gap_size > 30:
                gaps.append((gap_start / fps, len(detection_mask) / fps, gap_size))

        for start, end, size in gaps:
            mid = (start + end) / 2
            ax.annotate(f'{size}', xy=(mid, 0.5), xytext=(mid, 1.5),
                        ha='center', va='bottom', fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))

        ax.set_xlim([0, extent_end])
        ax.set_ylim([0, 2 if gaps else 1])
        ax.set_xlabel('Time (seconds)')
        ax.set_yticks([])
        ax.set_title('Detection Presence', fontsize=10)

        coverage_pct = stage_stats[stage]['coverage_percent']
        ax.text(0.02, 0.5, f'{coverage_pct:.1f}%', transform=ax.transAxes,
                fontsize=10, va='center', weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Row 3: Rolling coverage
    window = 100
    for idx, (stage, data) in enumerate(dataset_items):
        ax = fig.add_subplot(gs[2, idx])

        frame_counts = data['frame_counts']
        if frame_counts.size == 0:
            ax.text(0.5, 0.5, 'No frames', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        detection_mask = frame_counts > 0
        rolling_coverage = np.convolve(
            detection_mask, np.ones(window) / max(window, 1), mode='same'
        ) * 100
        time_seconds = np.arange(len(detection_mask)) / max(fps, 1e-6)

        ax.fill_between(time_seconds, 0, rolling_coverage,
                        color=stage_colors[stage], alpha=0.3)
        ax.plot(time_seconds, rolling_coverage,
                color=stage_colors[stage], alpha=0.8, linewidth=1)

        in_gap = False
        gap_start = 0.0
        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_start = i / fps
                in_gap = True
            elif has_det and in_gap:
                ax.axvspan(gap_start, i / fps, color='red', alpha=0.1)
                in_gap = False
        if in_gap:
            ax.axvspan(gap_start, len(detection_mask) / fps, color='red', alpha=0.1)

        ax.set_ylim([0, 105])
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'Rolling Coverage (window={window} frames)', fontsize=10)
        ax.grid(True, alpha=0.3)

        mean_cov = float(rolling_coverage.mean()) if rolling_coverage.size else 0.0
        ax.axhline(y=mean_cov, color='black', linestyle='--',
                   alpha=0.5, linewidth=1, label=f'Mean: {mean_cov:.1f}%')
        ax.legend(loc='lower right', fontsize=8)

    # Row 4: Gap analysis
    for idx, (stage, data) in enumerate(dataset_items):
        ax = fig.add_subplot(gs[3, idx])

        frame_counts = data['frame_counts']
        if frame_counts.size == 0:
            ax.text(0.5, 0.5, 'No frames', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        detection_mask = frame_counts > 0
        gap_sizes = []
        in_gap = False
        gap_start = 0

        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_start = i
                in_gap = True
            elif has_det and in_gap:
                gap_sizes.append(i - gap_start)
                in_gap = False
        if in_gap:
            gap_sizes.append(len(detection_mask) - gap_start)

        if gap_sizes:
            max_gap = max(gap_sizes)
            bins = np.arange(0, min(max_gap + 2, 50), 1)
            ax.hist(gap_sizes, bins=bins, color=stage_colors[stage], alpha=0.7, edgecolor='black')

            mean_gap = float(np.mean(gap_sizes))
            median_gap = float(np.median(gap_sizes))
            ax.axvline(x=mean_gap, color='red', linestyle='--', alpha=0.7,
                       label=f'Mean: {mean_gap:.1f}')
            ax.axvline(x=median_gap, color='orange', linestyle='--', alpha=0.7,
                       label=f'Median: {median_gap:.1f}')

            ax.set_xlabel('Gap Size (frames)')
            ax.set_ylabel('Count')
            ax.set_title(f'Gap Distribution | Total: {len(gap_sizes)} gaps', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            stats_text = (
                f"≤5: {sum(1 for g in gap_sizes if g <= 5)}\n"
                f"6-10: {sum(1 for g in gap_sizes if 5 < g <= 10)}\n"
                f">10: {sum(1 for g in gap_sizes if g > 10)}"
            )
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No gaps!', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='green', weight='bold')
            ax.set_xlabel('Gap Size (frames)')
            ax.set_ylabel('Count')
            ax.set_title('Gap Distribution', fontsize=10)
            ax.grid(True, alpha=0.3)

    title_suffix = ""
    if frame_range is not None:
        title_suffix = f" | Frames {frame_range[0]} - {frame_range[1]}"
    fig.suptitle(f'Refinement Pipeline Results - {refined_run}{title_suffix}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for stage_key in stage_order:
        stats = stage_stats[stage_key]
        print(f"{stage_names[stage_key]}:")
        print(f"  Coverage: {stats['coverage_percent']:.2f}% "
              f"({stats['frames_with_detections']} frames)")
        print(f"  Detections: {stats['total_detections']}")
        if stats['manual_edited_detections'] > 0:
            print(f"  Manual edited detections: {stats['manual_edited_detections']}")
        source_detection_summary = stats.get('source_detection_summary') or {}
        total_candidates = int(source_detection_summary.get('total_candidates', 0) or 0)
        if total_candidates > 0:
            print(f"  Source candidates: {total_candidates}")
            for label in ("accepted", "filtered", "duplicate", "manual_clear"):
                count = int(source_detection_summary.get(f"decision_{label}", 0) or 0)
                if count > 0:
                    print(f"  {label}: {count}")
    if 'filtered' in stage_stats:
        removed = stage_stats['original']['total_detections'] - stage_stats['filtered']['total_detections']
        print(f"\nDetections removed by filtering: {removed}")
    if 'interpolated' in stage_stats:
        baseline_stage = 'filtered' if 'filtered' in stage_stats else 'original'
        added = stage_stats['interpolated']['total_detections'] - stage_stats[baseline_stage]['total_detections']
        print(f"Detections added by interpolation: {added}")
    if 'curated' in stage_stats:
        print(f"Curated refined run used: {curated_detect_run_name}")


def render_refinement_pipeline_png(
    zarr_path: str,
    *,
    refined_run: Optional[str] = None,
    frame_range: Optional[Tuple[int, int]] = None,
    show_curated: bool = False,
    curated_detect_run: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Optional[str]]]:
    """
    Render refinement pipeline visualization to PNG bytes (no interactive window).

    Returns:
        (png_bytes, metadata)
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        # Suppress CLI-oriented print output during artifact rendering.
        with contextlib.redirect_stdout(io.StringIO()):
            visualize_refinement_pipeline(
                zarr_path=zarr_path,
                refined_run=refined_run,
                save_path=tmp_path,
                frame_range=frame_range,
                show=False,
                show_curated=show_curated,
                curated_detect_run=curated_detect_run,
            )
        if tmp_path is None or not os.path.exists(tmp_path):
            raise RuntimeError("refinement pipeline render did not produce a PNG file")
        png_bytes = Path(tmp_path).read_bytes()
        if not png_bytes:
            raise RuntimeError("refinement pipeline render produced an empty PNG file")
        return png_bytes, {
            "refined_run": refined_run,
            "curated_detect_run": curated_detect_run,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Visualize refinement pipeline results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize latest refined run
  %(prog)s data.zarr

  # Visualize specific refined run
  %(prog)s data.zarr --run refined_2025-10-03_21-39-43

  # Focus on specific frame window
  %(prog)s data.zarr --frames 1000 2000

  # Save visualization
  %(prog)s data.zarr --save refinement_viz.png
        """
    )

    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--run', '--refined-run', dest='refined_run',
                        help='Specific refined run to visualize (default: latest)')
    parser.add_argument('--save', help='Path to save the figure')
    parser.add_argument('--frames', nargs=2, type=int, metavar=('START', 'END'),
                        help='Frame range to visualize (inclusive start, exclusive end)')
    parser.add_argument('--list-runs', action='store_true',
                        help='List available refined runs and exit')
    parser.add_argument(
        '--show-curated',
        dest='show_curated',
        action='store_true',
        help='Include the canonical curated refined surface as an additional comparison stage when available.',
    )
    parser.add_argument(
        '--curated-detect-run',
        dest='curated_detect_run',
        help='Specific refined detect run to visualize when --show-curated is set.',
    )

    args = parser.parse_args()

    if args.list_runs:
        return summarize_refined_runs(args.zarr_path)

    frame_range = None
    if args.frames:
        start, end = args.frames
        if end <= start:
            raise ValueError("END frame must be greater than START frame.")
        frame_range = (start, end)

    visualize_refinement_pipeline(
        zarr_path=args.zarr_path,
        refined_run=args.refined_run,
        save_path=args.save,
        frame_range=frame_range,
        show_curated=args.show_curated,
        curated_detect_run=args.curated_detect_run,
    )

    return 0


if __name__ == '__main__':
    exit(main())
