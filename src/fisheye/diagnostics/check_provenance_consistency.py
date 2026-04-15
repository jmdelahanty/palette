#!/usr/bin/env python3
"""Diagnose provenance consistency across detect → crop → keypoint → arena assignment stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from ..shared.provenance_attrs import build_source_crop_snapshot_attrs
from ..shared.refined_detect_curation import (
    has_curated_refined_detect_surface,
    has_sparse_curated_refined_detect_instances_arrays,
)
from ..shared.type_conversions import as_int, normalize_attr


@dataclass
class ProvenanceRecord:
    detect_run: Optional[str]
    refined_run: Optional[str]
    crop_run: Optional[str]
    crop_source_path: Optional[str]
    crop_source_rows: Optional[int]
    keypoint_run: Optional[str]
    subject_mask_run: Optional[str]
    refined_subject_mask_run: Optional[str]
    arena_assignment_run: Optional[str]
    detect_rows: Optional[int]
    refined_rows: Optional[int]
    crop_rois: Optional[int]
    keypoint_rows: Optional[int]
    subject_mask_rows: Optional[int]
    refined_subject_mask_rows: Optional[int]
    arena_assignment_rows: Optional[int]
    issues: list[str]
    crop_source_drift_issues: list[str] = field(default_factory=list)
    downstream_crop_snapshot_issues: list[str] = field(default_factory=list)
    subject_mask_crop_snapshot_issues: list[str] = field(default_factory=list)
    refined_subject_mask_crop_snapshot_issues: list[str] = field(default_factory=list)


def _safe_len(arr: Optional[zarr.Array]) -> Optional[int]:
    if arr is None:
        return None
    try:
        return int(arr.shape[0])
    except Exception:
        return None


def _safe_get(group: Optional[zarr.Group], key: str) -> Optional[object]:
    if group is None:
        return None
    try:
        return group.get(key)
    except Exception:
        return None


def _group_for_path(root: zarr.Group, path: Optional[str]) -> Optional[zarr.Group]:
    if not path:
        return None
    normalized = path.strip("/")
    try:
        if normalized in root:
            return root[normalized]
    except Exception:
        return None
    return None


def _count_detections(group: Optional[zarr.Group]) -> Optional[int]:
    if group is None:
        return None
    if hasattr(group, "shape") and not hasattr(group, "get"):
        return _safe_len(group)  # type: ignore[arg-type]
    if isinstance(group, zarr.Array):
        return _safe_len(group)
    arr = _safe_get(group, "bbox_norm_coords") or _safe_get(group, "bbox_coords") or _safe_get(group, "bbox")
    return _safe_len(arr)


def _safe_array_data(group: Optional[zarr.Group], key: str) -> Optional[np.ndarray]:
    arr = _safe_get(group, key)
    if arr is None:
        return None
    try:
        return np.asarray(arr[:])
    except Exception:
        return None


def _latest(group: Optional[zarr.Group]) -> Optional[str]:
    if group is None:
        return None
    latest = group.attrs.get("latest")
    return latest if isinstance(latest, str) and latest in group else None


def _first_matching_run(group: Optional[zarr.Group], name: Optional[str]) -> Optional[zarr.Group]:
    if group is None or not name:
        return None
    return group.get(name)


def _normalize_label(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _legacy_sparse_refined_label(refined_group: zarr.Group) -> Optional[str]:
    manual_label = refined_group.attrs.get("manual_review_latest")
    if isinstance(manual_label, str) and manual_label in refined_group:
        return manual_label
    for candidate in ("manual", "interpolated", "filtered"):
        if candidate in refined_group:
            return candidate
    return None


def _canonical_refined_surface_path(
    refined_parent_name: str,
    refined_run_name: str,
    refined_group: zarr.Group,
) -> Optional[str]:
    if has_sparse_curated_refined_detect_instances_arrays(refined_group):
        return f"{refined_parent_name}/{refined_run_name}/instances"
    if has_curated_refined_detect_surface(refined_group):
        return f"{refined_parent_name}/{refined_run_name}"
    return None


def _preferred_refined_rowset_path(
    refined_parent_name: str,
    refined_run_name: str,
    refined_group: zarr.Group,
) -> Optional[str]:
    canonical = _canonical_refined_surface_path(
        refined_parent_name,
        refined_run_name,
        refined_group,
    )
    if canonical is not None:
        return canonical
    legacy_label = _legacy_sparse_refined_label(refined_group)
    if legacy_label is not None:
        return f"{refined_parent_name}/{refined_run_name}/{legacy_label}"
    return None


def _expected_crop_source_path(
    *,
    refined_parent_name: str,
    refined_run_name: str,
    refined_group: zarr.Group,
    detect_latest: Optional[str],
) -> Optional[str]:
    review_status = refined_group.attrs.get("detect_review_status")
    resolved_group = None
    if isinstance(review_status, dict):
        resolved_group = _normalize_label(
            review_status.get("target_group") or review_status.get("resolved_group")
        )

    if resolved_group in {"raw", "detect"}:
        if detect_latest:
            return f"detect_runs/{detect_latest}"
        source_detect = refined_group.attrs.get("source_detect_run")
        if source_detect:
            return f"detect_runs/{source_detect}"
        return None

    if resolved_group in {"refined", "instances"}:
        canonical = _canonical_refined_surface_path(
            refined_parent_name,
            refined_run_name,
            refined_group,
        )
        if canonical is not None:
            return canonical

    if resolved_group == "manual":
        manual_label = _legacy_sparse_refined_label(refined_group)
        if manual_label is not None and manual_label.startswith("manual"):
            return f"{refined_parent_name}/{refined_run_name}/{manual_label}"

    if resolved_group and resolved_group in refined_group:
        return f"{refined_parent_name}/{refined_run_name}/{resolved_group}"

    return _preferred_refined_rowset_path(
        refined_parent_name,
        refined_run_name,
        refined_group,
    )


def _format_drift_issue(
    *,
    crop_run: Optional[str],
    source_path: Optional[str],
    field_name: str,
    changed_rows: int,
    changed_frames: Optional[int] = None,
) -> str:
    run_label = f"Crop run '{crop_run}'" if crop_run else "Crop run"
    source_label = f"upstream source '{source_path}'" if source_path else "upstream source"
    detail = f"{field_name} differ for {changed_rows} row(s)"
    if changed_frames is not None:
        detail += f" across {changed_frames} frame(s)"
    return f"{run_label} snapshot drifted from {source_label}: {detail}."


def _collect_crop_source_drift_issues(
    *,
    crop_run: Optional[str],
    crop_group: zarr.Group,
    source_group: Optional[zarr.Group],
    source_path: Optional[str],
) -> list[str]:
    if source_group is None:
        return []

    issues: list[str] = []
    crop_source_drift_issues: list[str] = []
    crop_frame_indices = _safe_array_data(crop_group, "frame_indices")
    source_frame_indices = _safe_array_data(source_group, "frame_indices")
    if crop_frame_indices is not None and source_frame_indices is not None:
        if crop_frame_indices.shape != source_frame_indices.shape:
            issues.append(
                _format_drift_issue(
                    crop_run=crop_run,
                    source_path=source_path,
                    field_name="frame_indices shapes",
                    changed_rows=abs(int(np.prod(crop_frame_indices.shape or (0,))) - int(np.prod(source_frame_indices.shape or (0,)))),
                )
            )
        else:
            changed_rows = np.where(
                np.asarray(crop_frame_indices, dtype=np.int64).reshape(-1)
                != np.asarray(source_frame_indices, dtype=np.int64).reshape(-1)
            )[0]
            if changed_rows.size:
                changed_frames = np.unique(
                    np.concatenate(
                        [
                            np.asarray(crop_frame_indices, dtype=np.int64).reshape(-1)[changed_rows],
                            np.asarray(source_frame_indices, dtype=np.int64).reshape(-1)[changed_rows],
                        ]
                    )
                )
                issues.append(
                    _format_drift_issue(
                        crop_run=crop_run,
                        source_path=source_path,
                        field_name="frame_indices",
                        changed_rows=int(changed_rows.size),
                        changed_frames=int(changed_frames.size),
                    )
                )

    crop_bbox = _safe_array_data(crop_group, "bbox_norm_coords")
    source_bbox = _safe_array_data(source_group, "bbox_norm_coords")
    if crop_bbox is not None and source_bbox is not None:
        if crop_bbox.shape != source_bbox.shape:
            issues.append(
                f"Crop run '{crop_run}' snapshot drifted from upstream source '{source_path}': "
                f"bbox_norm_coords shape {tuple(crop_bbox.shape)} != {tuple(source_bbox.shape)}."
            )
        else:
            bbox_drift_mask = ~np.isclose(
                np.asarray(crop_bbox, dtype=np.float64),
                np.asarray(source_bbox, dtype=np.float64),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            )
            changed_bbox_rows = np.where(np.any(bbox_drift_mask, axis=1))[0]
            if changed_bbox_rows.size:
                changed_frames = None
                if source_frame_indices is not None and source_frame_indices.shape[0] == source_bbox.shape[0]:
                    changed_frames = int(
                        np.unique(
                            np.asarray(source_frame_indices, dtype=np.int64).reshape(-1)[changed_bbox_rows]
                        ).size
                    )
                issues.append(
                    _format_drift_issue(
                        crop_run=crop_run,
                        source_path=source_path,
                        field_name="bbox_norm_coords",
                        changed_rows=int(changed_bbox_rows.size),
                        changed_frames=changed_frames,
                    )
                )

    return issues


def _infer_crop_storage_mode(crop_group: zarr.Group) -> str:
    stored = normalize_attr(crop_group.attrs.get("crop_storage_mode"))
    if stored:
        return stored
    return "materialized" if _safe_get(crop_group, "roi_images") is not None else "geometry_only"


def _compare_snapshot_value(field_name: str, value: object) -> object:
    if field_name == "source_crop_revision":
        numeric = as_int(value)
        return int(numeric) if numeric is not None and numeric >= 0 else None
    return normalize_attr(value)


def _collect_downstream_crop_snapshot_issues(
    *,
    stage_label: str,
    run_name: Optional[str],
    run_group: Optional[zarr.Group],
    latest_crop: Optional[str],
    crop_group: Optional[zarr.Group],
) -> list[str]:
    if run_group is None or crop_group is None:
        return []

    issues: list[str] = []
    source_crop_run = normalize_attr(run_group.attrs.get("source_crop_run"))
    run_label = f"{stage_label} run '{run_name}'" if run_name else f"{stage_label} run"
    crop_label = f"crop run '{latest_crop}'" if latest_crop else "current crop run"

    if not source_crop_run:
        issues.append(f"{run_label} is missing source_crop_run provenance.")
        return issues

    if latest_crop and source_crop_run != latest_crop:
        issues.append(
            f"{run_label} references crop '{source_crop_run}', but latest crop is '{latest_crop}'."
        )
        return issues

    expected_snapshot = build_source_crop_snapshot_attrs(
        crop_group.attrs,
        source_crop_storage_mode=_infer_crop_storage_mode(crop_group),
    )
    if not expected_snapshot:
        return issues

    mismatches: list[str] = []
    for field_name, expected_value in expected_snapshot.items():
        actual_value = _compare_snapshot_value(field_name, run_group.attrs.get(field_name))
        expected_comp = _compare_snapshot_value(field_name, expected_value)
        if expected_comp is None:
            continue
        if actual_value is None:
            mismatches.append(f"missing {field_name}")
        elif actual_value != expected_comp:
            mismatches.append(f"{field_name}={actual_value!r} expected {expected_comp!r}")

    if mismatches:
        issues.append(f"{run_label} crop snapshot drifted from {crop_label}: {'; '.join(mismatches)}.")

    return issues


def _collect_provenance(root: zarr.Group) -> ProvenanceRecord:
    issues: list[str] = []

    detect_parent = root.get("detect_runs")
    detect_latest = _latest(detect_parent)
    detect_group = _first_matching_run(detect_parent, detect_latest)
    detect_rows = _count_detections(detect_group)

    refined_parent_name = "refined_detect_runs"
    refined_parent = root.get(refined_parent_name)
    if refined_parent is None:
        refined_parent_name = "refined_runs"
        refined_parent = root.get(refined_parent_name)
    refined_latest = _latest(refined_parent)
    refined_group = _first_matching_run(refined_parent, refined_latest)
    refined_rows = None
    if refined_group is not None:
        refined_source_path = _preferred_refined_rowset_path(
            refined_parent_name,
            refined_latest,
            refined_group,
        ) if refined_latest else None
        if refined_source_path is not None:
            refined_rows = _count_detections(_group_for_path(root, refined_source_path))
            if refined_rows is None:
                issues.append(
                    f"Refined detect run '{refined_latest}' is missing detection arrays under "
                    f"{refined_source_path} "
                    "(expected one of: bbox_norm_coords, bbox_coords, bbox)."
                )
        source_detect = refined_group.attrs.get("source_detect_run")
        if detect_latest and source_detect and detect_latest != source_detect:
            issues.append(
                f"Refined detect run '{refined_latest}' references detect '{source_detect}', "
                f"but latest detect is '{detect_latest}'."
            )

    crop_parent = root.get("crop_runs")
    crop_latest = _latest(crop_parent)
    crop_group = _first_matching_run(crop_parent, crop_latest)
    crop_rois = _safe_len(_safe_get(crop_group, "roi_images")) if crop_group else None
    crop_source_rows = None
    crop_source_path = None
    crop_source_drift_issues: list[str] = []
    if crop_group is not None:
        crop_source = crop_group.attrs.get("detection_source_path")
        crop_source_path = crop_source

        def _expected_crop_path() -> Optional[str]:
            if refined_group is None:
                if detect_latest:
                    return f"detect_runs/{detect_latest}"
                return None

            return _expected_crop_source_path(
                refined_parent_name=refined_parent_name,
                refined_run_name=refined_latest,
                refined_group=refined_group,
                detect_latest=detect_latest,
            )

        expected_path = _expected_crop_path()
        source_path = crop_source or expected_path
        if source_path:
            crop_source_path = source_path
        crop_source_group = _group_for_path(root, source_path)
        crop_source_rows = _count_detections(crop_source_group)
        if crop_source and expected_path and crop_source != expected_path:
            issues.append(
                f"Crop run '{crop_latest}' sourced from '{crop_source}' but expected '{expected_path}'."
            )
        crop_source_drift_issues = _collect_crop_source_drift_issues(
            crop_run=crop_latest,
            crop_group=crop_group,
            source_group=crop_source_group,
            source_path=source_path,
        )
        issues.extend(crop_source_drift_issues)

    keypoint_parent = root.get("refined_keypoints_runs") or root.get("keypoints_runs")
    keypoint_latest = _latest(keypoint_parent)
    keypoint_group = _first_matching_run(keypoint_parent, keypoint_latest)
    keypoint_rows = _safe_len(_safe_get(keypoint_group, "heading")) if keypoint_group else None
    downstream_crop_snapshot_issues: list[str] = []
    downstream_crop_snapshot_issues.extend(
        _collect_downstream_crop_snapshot_issues(
            stage_label="Keypoint",
            run_name=keypoint_latest,
            run_group=keypoint_group,
            latest_crop=crop_latest,
            crop_group=crop_group,
        )
    )

    eye_mask_parent = root.get("eye_masks_runs")
    eye_mask_latest = _latest(eye_mask_parent)
    eye_mask_group = _first_matching_run(eye_mask_parent, eye_mask_latest)
    downstream_crop_snapshot_issues.extend(
        _collect_downstream_crop_snapshot_issues(
            stage_label="Eye mask",
            run_name=eye_mask_latest,
            run_group=eye_mask_group,
            latest_crop=crop_latest,
            crop_group=crop_group,
        )
    )
    issues.extend(downstream_crop_snapshot_issues)

    subject_mask_parent = root.get("subject_mask_runs")
    subject_mask_latest = _latest(subject_mask_parent)
    subject_mask_group = _first_matching_run(subject_mask_parent, subject_mask_latest)
    subject_mask_rows = _safe_len(_safe_get(subject_mask_group, "masks_roi")) if subject_mask_group else None
    if subject_mask_rows is None and subject_mask_group is not None:
        subject_mask_rows = _safe_len(_safe_get(subject_mask_group, "mask_probs_roi"))
    subject_mask_crop_snapshot_issues = _collect_downstream_crop_snapshot_issues(
        stage_label="Subject mask",
        run_name=subject_mask_latest,
        run_group=subject_mask_group,
        latest_crop=crop_latest,
        crop_group=crop_group,
    )
    issues.extend(subject_mask_crop_snapshot_issues)

    refined_subject_mask_parent = root.get("refined_subject_masks_runs")
    refined_subject_mask_latest = _latest(refined_subject_mask_parent)
    refined_subject_mask_group = _first_matching_run(refined_subject_mask_parent, refined_subject_mask_latest)
    refined_subject_mask_rows = (
        _safe_len(_safe_get(refined_subject_mask_group, "masks_roi")) if refined_subject_mask_group else None
    )
    if refined_subject_mask_rows is None and refined_subject_mask_group is not None:
        refined_subject_mask_rows = _safe_len(_safe_get(refined_subject_mask_group, "mask_probs_roi"))
    refined_subject_mask_crop_snapshot_issues = _collect_downstream_crop_snapshot_issues(
        stage_label="Refined subject mask",
        run_name=refined_subject_mask_latest,
        run_group=refined_subject_mask_group,
        latest_crop=crop_latest,
        crop_group=crop_group,
    )
    issues.extend(refined_subject_mask_crop_snapshot_issues)

    arena_parent = root.get("arena_assignment_runs")
    arena_latest = _latest(arena_parent)
    arena_group = _first_matching_run(arena_parent, arena_latest)
    arena_rows = _safe_len(_safe_get(arena_group, "arena_ids")) if arena_group else None
    arena_source_rows = None
    if arena_group is not None:
        arena_source_detect = arena_group.attrs.get("source_detect_run")
        if detect_latest and arena_source_detect and detect_latest != arena_source_detect:
            issues.append(
                f"Arena assignment run '{arena_latest}' references detect '{arena_source_detect}' but latest detect is '{detect_latest}'."
            )
        arena_source_refined = arena_group.attrs.get("source_refined_run")
        if refined_latest and arena_source_refined and refined_latest != arena_source_refined:
            issues.append(
                f"Arena assignment run '{arena_latest}' references refined detect '{arena_source_refined}' but latest refined detect is '{refined_latest}'."
            )
        if arena_source_refined:
            arena_refined_group = _first_matching_run(refined_parent, arena_source_refined)
            arena_refined_path = (
                _preferred_refined_rowset_path(
                    refined_parent_name,
                    arena_source_refined,
                    arena_refined_group,
                )
                if arena_refined_group is not None
                else None
            )
            if arena_refined_path is not None:
                arena_source_rows = _count_detections(_group_for_path(root, arena_refined_path))
        if arena_source_rows is None and arena_source_detect:
            arena_source_rows = _count_detections(_group_for_path(root, f"detect_runs/{arena_source_detect}"))

    if crop_rois is not None and crop_source_rows is not None and crop_rois != crop_source_rows:
        issues.append(
            f"Crop ROI count ({crop_rois}) != detection source count ({crop_source_rows})"
            + (f" from '{crop_source_path}'." if crop_source_path else ".")
        )
    if crop_rois is not None and keypoint_rows is not None and crop_rois != keypoint_rows:
        issues.append(
            f"Crop ROI count ({crop_rois}) != keypoint heading count ({keypoint_rows})."
        )
    if arena_rows is not None and arena_source_rows is not None and arena_rows != arena_source_rows:
        issues.append(
            f"Arena assignments ({arena_rows}) != detection source count ({arena_source_rows})."
        )

    return ProvenanceRecord(
        detect_run=detect_latest,
        refined_run=refined_latest,
        crop_run=crop_latest,
        crop_source_path=crop_source_path,
        crop_source_rows=crop_source_rows,
        keypoint_run=keypoint_latest,
        subject_mask_run=subject_mask_latest,
        refined_subject_mask_run=refined_subject_mask_latest,
        arena_assignment_run=arena_latest,
        detect_rows=detect_rows,
        refined_rows=refined_rows,
        crop_rois=crop_rois,
        keypoint_rows=keypoint_rows,
        subject_mask_rows=subject_mask_rows,
        refined_subject_mask_rows=refined_subject_mask_rows,
        arena_assignment_rows=arena_rows,
        issues=issues,
        crop_source_drift_issues=crop_source_drift_issues,
        downstream_crop_snapshot_issues=downstream_crop_snapshot_issues,
        subject_mask_crop_snapshot_issues=subject_mask_crop_snapshot_issues,
        refined_subject_mask_crop_snapshot_issues=refined_subject_mask_crop_snapshot_issues,
    )


def collect_provenance(root: zarr.Group) -> ProvenanceRecord:
    """Collect provenance consistency info without printing."""
    return _collect_provenance(root)


def show_provenance(zarr_path: Path) -> None:
    console = Console()
    root = zarr.open(str(zarr_path), mode="r")

    record = _collect_provenance(root)

    table = Table(title="Provenance Summary", show_lines=False, box=None)
    table.add_column("Stage", style="cyan")
    table.add_column("Run")
    table.add_column("Rows", justify="right")

    table.add_row("Detect", record.detect_run or "—", str(record.detect_rows or "—"))
    table.add_row("Refined Detect", record.refined_run or "—", str(record.refined_rows or "—"))
    table.add_row("Crop", record.crop_run or "—", str(record.crop_rois or "—"))
    table.add_row("Keypoints", record.keypoint_run or "—", str(record.keypoint_rows or "—"))
    table.add_row("Subject Masks", record.subject_mask_run or "—", str(record.subject_mask_rows or "—"))
    table.add_row(
        "Refined Subject Masks",
        record.refined_subject_mask_run or "—",
        str(record.refined_subject_mask_rows or "—"),
    )
    table.add_row("Arena Assignment", record.arena_assignment_run or "—", str(record.arena_assignment_rows or "—"))

    console.print(table)
    source_label = record.crop_source_path or "—"
    det_count = record.crop_source_rows if record.crop_source_rows is not None else "—"
    crop_count = record.crop_rois if record.crop_rois is not None else "—"
    kp_count = record.keypoint_rows if record.keypoint_rows is not None else "—"
    subject_mask_count = record.subject_mask_rows if record.subject_mask_rows is not None else "—"
    refined_subject_mask_count = record.refined_subject_mask_rows if record.refined_subject_mask_rows is not None else "—"
    console.print(
        f"[dim]Lineage counts:[/dim] source={source_label} det={det_count} "
        f"crop={crop_count} kpt={kp_count} subject={subject_mask_count} refined_subject={refined_subject_mask_count}"
    )

    if record.issues:
        console.print("[bold red]Inconsistencies detected:[/bold red]")
        for issue in record.issues:
            console.print(f"  • {issue}")
    else:
        console.print("[green]No provenance issues detected.[/green]")


def parse_args(argv: Optional[sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check detect/crop/keypoint/ID provenance consistency."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive")
    return parser.parse_args(argv)


def main(argv: Optional[sequence[str]] = None) -> None:
    args = parse_args(argv)
    show_provenance(args.zarr_path)


if __name__ == "__main__":  # pragma: no cover
    main()
