"""Subject-body refined-mask quality checks.

This module owns mask-local QC for the refined ``subject_body`` component. It
does not edit mask pixels. It computes deterministic topology/shape flags that
review tools and downstream geometry stages can consume before trusting body
centerlines or splines.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import convolve
from skimage.measure import label as label_components
from skimage.morphology import convex_hull_image, skeletonize
import zarr

from ..shared.detect_reason_codec import write_reason_columns
from ..shared.json_safety import json_attr_safe
from ..shared.mask_geometry import hole_stats
from ..shared.mask_store import MaskStoreError, open_mask_store
from ..shared.refined_subject_mask_mutation import (
    resolve_mutable_refined_subject_mask_run,
)
from ..shared.subject_mask_chunks import refined_subject_mask_metric_row_chunk
from ..shared.zarr_io import open_zarr_root

SUBJECT_BODY_MASK_QC_SCHEMA_ID = "refined_subject_body_mask_qc"
SUBJECT_BODY_MASK_QC_SCHEMA_VERSION = 1
SUBJECT_BODY_MASK_QC_METHOD = "subject_body_mask_qc_v1"
SUBJECT_BODY_MASK_QC_METHOD_VERSION = 1
SUBJECT_BODY_COMPONENT = "subject_body"


@dataclass(frozen=True)
class SubjectBodyMaskQcPolicy:
    min_area_px: float = 1.0
    max_area_px: Optional[float] = None
    max_component_count: int = 1
    min_largest_component_fraction: float = 0.95
    min_solidity: Optional[float] = 0.15
    max_hole_area_fraction: float = 0.01
    max_skeleton_endpoint_count: int = 4
    max_skeleton_branchpoint_count: int = 6
    max_thin_spur_score: float = 8.0


@dataclass(frozen=True)
class SubjectBodyMaskQcBatch:
    component_count: np.ndarray
    largest_component_area_px: np.ndarray
    total_area_px: np.ndarray
    largest_component_fraction: np.ndarray
    bbox_width_px: np.ndarray
    bbox_height_px: np.ndarray
    bbox_aspect_ratio: np.ndarray
    solidity: np.ndarray
    hole_area_px: np.ndarray
    hole_area_fraction: np.ndarray
    skeleton_endpoint_count: np.ndarray
    skeleton_branchpoint_count: np.ndarray
    thin_spur_score: np.ndarray
    severe_qc_failure: np.ndarray
    requires_review: np.ndarray
    reason_labels: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_json_safe = json_attr_safe


def _policy_payload(policy: SubjectBodyMaskQcPolicy) -> dict[str, object]:
    return {
        "min_area_px": float(policy.min_area_px),
        "max_area_px": None if policy.max_area_px is None else float(policy.max_area_px),
        "max_component_count": int(policy.max_component_count),
        "min_largest_component_fraction": float(policy.min_largest_component_fraction),
        "min_solidity": None if policy.min_solidity is None else float(policy.min_solidity),
        "max_hole_area_fraction": float(policy.max_hole_area_fraction),
        "max_skeleton_endpoint_count": int(policy.max_skeleton_endpoint_count),
        "max_skeleton_branchpoint_count": int(policy.max_skeleton_branchpoint_count),
        "max_thin_spur_score": float(policy.max_thin_spur_score),
    }


def _component_sizes(mask: np.ndarray) -> tuple[int, np.ndarray]:
    labels = label_components(np.asarray(mask, dtype=bool), connectivity=2)
    count = int(labels.max()) if labels.size else 0
    if count <= 0:
        return 0, np.zeros((0,), dtype=np.int64)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)[1:]
    return count, np.asarray(sizes, dtype=np.int64)


def _foreground_crop(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask_bool)
    if int(xs.size) == 0:
        return mask_bool
    cropped = mask_bool[int(ys.min()): int(ys.max()) + 1, int(xs.min()): int(xs.max()) + 1]
    return np.ascontiguousarray(cropped)


def _bbox_metrics(mask: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if int(xs.size) == 0:
        return np.nan, np.nan, np.nan
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    aspect = float(width / height) if height > 0.0 else np.nan
    return width, height, aspect


def _solidity(mask: np.ndarray, total_area: float) -> float:
    if total_area <= 0.0:
        return 0.0
    try:
        hull = convex_hull_image(np.asarray(mask, dtype=bool))
    except Exception:
        return np.nan
    hull_area = float(np.count_nonzero(hull))
    if hull_area <= 0.0:
        return 0.0
    return float(total_area / hull_area)


def _hole_metrics(mask: np.ndarray, total_area: float) -> tuple[float, float]:
    if total_area <= 0.0:
        return 0.0, 0.0
    _hole_count, fraction, hole_area = hole_stats(mask)
    return hole_area, fraction


def _skeleton_counts(mask: np.ndarray) -> tuple[int, int]:
    mask_bool = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    if int(np.count_nonzero(mask_bool)) == 0:
        return 0, 0
    skeleton = skeletonize(mask_bool)
    if int(np.count_nonzero(skeleton)) == 0:
        return 0, 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0) - skeleton.astype(np.uint8)
    endpoints = int(np.count_nonzero(skeleton & (neighbor_count == 1)))
    branchpoints = int(np.count_nonzero(skeleton & (neighbor_count > 2)))
    return endpoints, branchpoints


def compute_subject_body_mask_qc(
    masks_roi: np.ndarray,
    *,
    policy: SubjectBodyMaskQcPolicy | None = None,
) -> SubjectBodyMaskQcBatch:
    """Compute per-row subject-body mask QC metrics and reason tags."""

    resolved_policy = policy or SubjectBodyMaskQcPolicy()
    masks = np.asarray(masks_roi, dtype=np.uint8)
    if masks.ndim != 3:
        raise ValueError(f"masks_roi must be 3D (N, H, W), got shape {tuple(masks.shape)}.")

    row_count = int(masks.shape[0])
    component_count = np.zeros((row_count,), dtype=np.int32)
    largest_component_area_px = np.zeros((row_count,), dtype=np.float32)
    total_area_px = np.zeros((row_count,), dtype=np.float32)
    largest_component_fraction = np.zeros((row_count,), dtype=np.float32)
    bbox_width_px = np.full((row_count,), np.nan, dtype=np.float32)
    bbox_height_px = np.full((row_count,), np.nan, dtype=np.float32)
    bbox_aspect_ratio = np.full((row_count,), np.nan, dtype=np.float32)
    solidity = np.full((row_count,), np.nan, dtype=np.float32)
    hole_area_px = np.zeros((row_count,), dtype=np.float32)
    hole_area_fraction = np.zeros((row_count,), dtype=np.float32)
    skeleton_endpoint_count = np.zeros((row_count,), dtype=np.int32)
    skeleton_branchpoint_count = np.zeros((row_count,), dtype=np.int32)
    thin_spur_score = np.zeros((row_count,), dtype=np.float32)
    severe_qc_failure = np.zeros((row_count,), dtype=bool)
    requires_review = np.zeros((row_count,), dtype=bool)
    reason_labels = np.full((row_count,), "ok", dtype=object)

    for row_idx in range(row_count):
        mask = masks[int(row_idx)] > 0
        total_area = float(np.count_nonzero(mask))
        work_mask = _foreground_crop(mask) if total_area > 0.0 else mask
        count, sizes = _component_sizes(work_mask)
        largest_area = float(sizes.max()) if int(sizes.size) else 0.0
        largest_fraction = float(largest_area / total_area) if total_area > 0.0 else 0.0
        width, height, aspect = _bbox_metrics(mask)
        sol = _solidity(work_mask, total_area)
        holes, hole_fraction = _hole_metrics(work_mask, total_area)
        endpoints, branchpoints = _skeleton_counts(work_mask)
        spur_score = float(max(0, endpoints - 2) + max(0, branchpoints))

        component_count[row_idx] = np.int32(count)
        largest_component_area_px[row_idx] = np.float32(largest_area)
        total_area_px[row_idx] = np.float32(total_area)
        largest_component_fraction[row_idx] = np.float32(largest_fraction)
        bbox_width_px[row_idx] = np.float32(width)
        bbox_height_px[row_idx] = np.float32(height)
        bbox_aspect_ratio[row_idx] = np.float32(aspect)
        solidity[row_idx] = np.float32(sol)
        hole_area_px[row_idx] = np.float32(holes)
        hole_area_fraction[row_idx] = np.float32(hole_fraction)
        skeleton_endpoint_count[row_idx] = np.int32(endpoints)
        skeleton_branchpoint_count[row_idx] = np.int32(branchpoints)
        thin_spur_score[row_idx] = np.float32(spur_score)

        tags: list[str] = []
        severe = False
        if total_area <= 0.0:
            tags.append("missing_subject_body_mask")
            severe = True
        elif total_area < float(resolved_policy.min_area_px):
            tags.append("small_body_mask")
            severe = True
        if resolved_policy.max_area_px is not None and total_area > float(resolved_policy.max_area_px):
            tags.append("large_body_mask")
        if count > int(resolved_policy.max_component_count):
            tags.append("fragmented_subject_body_mask")
            severe = True
        if total_area > 0.0 and largest_fraction < float(resolved_policy.min_largest_component_fraction):
            tags.append("low_largest_component_fraction")
            severe = True
        if (
            total_area > 0.0
            and resolved_policy.min_solidity is not None
            and np.isfinite(sol)
            and sol < float(resolved_policy.min_solidity)
        ):
            tags.append("low_solidity")
        if hole_fraction > float(resolved_policy.max_hole_area_fraction):
            tags.append("large_hole_area")
        if endpoints > int(resolved_policy.max_skeleton_endpoint_count):
            tags.append("excess_body_skeleton_endpoints")
            severe = True
        if branchpoints > int(resolved_policy.max_skeleton_branchpoint_count):
            tags.append("branched_body_mask")
            severe = True
        if spur_score > float(resolved_policy.max_thin_spur_score):
            tags.append("thin_attached_artifact")
            severe = True

        if tags:
            tags.append("requires_review")
        severe_qc_failure[row_idx] = bool(severe)
        requires_review[row_idx] = bool(tags)
        reason_labels[row_idx] = "|".join(tags) if tags else "ok"

    return SubjectBodyMaskQcBatch(
        component_count=component_count,
        largest_component_area_px=largest_component_area_px,
        total_area_px=total_area_px,
        largest_component_fraction=largest_component_fraction,
        bbox_width_px=bbox_width_px,
        bbox_height_px=bbox_height_px,
        bbox_aspect_ratio=bbox_aspect_ratio,
        solidity=solidity,
        hole_area_px=hole_area_px,
        hole_area_fraction=hole_area_fraction,
        skeleton_endpoint_count=skeleton_endpoint_count,
        skeleton_branchpoint_count=skeleton_branchpoint_count,
        thin_spur_score=thin_spur_score,
        severe_qc_failure=severe_qc_failure,
        requires_review=requires_review,
        reason_labels=reason_labels,
    )


def _create_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Sequence[int],
    dtype: object,
    chunks: Sequence[int],
    fill_value: object = 0,
) -> None:
    if name in group:
        del group[name]
    group.create_array(
        name,
        shape=tuple(int(dim) for dim in shape),
        dtype=dtype,
        chunks=tuple(int(dim) for dim in chunks),
        fill_value=fill_value,
        overwrite=True,
    )


def _reason_counts(labels: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in np.asarray(labels, dtype=object).reshape(-1):
        text = str(label)
        counts[text] = int(counts.get(text, 0)) + 1
    return dict(sorted(counts.items()))


def _resolve_refined_run(root: zarr.Group, refined_run: Optional[str]) -> tuple[str, zarr.Group]:
    parent = root.get("refined_subject_masks_runs")
    if parent is None:
        raise ValueError("Archive has no refined_subject_masks_runs group.")
    if refined_run:
        run_name = str(refined_run)
    else:
        latest = parent.attrs.get("latest")
        if latest is None:
            keys = sorted(str(key) for key in parent.keys())
            if not keys:
                raise ValueError("refined_subject_masks_runs has no runs.")
            run_name = keys[-1]
        else:
            run_name = str(latest)
    if run_name not in parent:
        raise ValueError(f"refined_subject_masks_runs/{run_name} not found.")
    return run_name, parent[run_name]


def _subject_body_channel_index(refined_group: zarr.Group) -> int:
    labels = refined_group.attrs.get("mask_labels")
    if not isinstance(labels, (list, tuple)):
        raise ValueError("refined subject-mask run is missing mask_labels.")
    for idx, label in enumerate(labels):
        if str(label) == SUBJECT_BODY_COMPONENT:
            return int(idx)
    raise ValueError("refined subject-mask run does not include subject_body in mask_labels.")


def _available_channel(refined_group: zarr.Group, component_idx: int) -> bool:
    available = refined_group.get("available_channels")
    if available is None:
        return True
    values = np.asarray(available[:], dtype=bool).reshape(-1)
    return int(component_idx) < int(values.shape[0]) and bool(values[int(component_idx)])


def _prepare_qc_group(
    refined_group: zarr.Group,
    *,
    row_count: int,
    policy: SubjectBodyMaskQcPolicy,
    overwrite: bool,
) -> zarr.Group:
    components = refined_group.require_group("components")
    component_group = components.require_group(SUBJECT_BODY_COMPONENT)
    if "qc" in component_group:
        if not overwrite:
            raise ValueError(
                "components/subject_body/qc already exists. Pass overwrite=True or --overwrite to replace it."
            )
        del component_group["qc"]
    qc_group = component_group.create_group("qc")
    chunks_1d = (refined_subject_mask_metric_row_chunk(row_count),)
    for name, dtype, fill in (
        ("component_count", np.int32, 0),
        ("largest_component_area_px", np.float32, np.nan),
        ("total_area_px", np.float32, np.nan),
        ("largest_component_fraction", np.float32, np.nan),
        ("bbox_width_px", np.float32, np.nan),
        ("bbox_height_px", np.float32, np.nan),
        ("bbox_aspect_ratio", np.float32, np.nan),
        ("solidity", np.float32, np.nan),
        ("hole_area_px", np.float32, np.nan),
        ("hole_area_fraction", np.float32, np.nan),
        ("skeleton_endpoint_count", np.int32, 0),
        ("skeleton_branchpoint_count", np.int32, 0),
        ("thin_spur_score", np.float32, 0.0),
        ("severe_qc_failure", bool, False),
        ("requires_review", bool, False),
    ):
        _create_array(qc_group, name, shape=(row_count,), dtype=dtype, chunks=chunks_1d, fill_value=fill)
    qc_group.attrs.update(
        {
            "schema_id": SUBJECT_BODY_MASK_QC_SCHEMA_ID,
            "schema_version": SUBJECT_BODY_MASK_QC_SCHEMA_VERSION,
            "method": SUBJECT_BODY_MASK_QC_METHOD,
            "method_version": SUBJECT_BODY_MASK_QC_METHOD_VERSION,
            "component_name": SUBJECT_BODY_COMPONENT,
            "row_axis": "refined_subject_mask_rows",
            "policy": _policy_payload(policy),
        }
    )
    return qc_group


def _write_qc_batch(qc_group: zarr.Group, row_slice: slice, batch: SubjectBodyMaskQcBatch) -> None:
    for name in (
        "component_count",
        "largest_component_area_px",
        "total_area_px",
        "largest_component_fraction",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_aspect_ratio",
        "solidity",
        "hole_area_px",
        "hole_area_fraction",
        "skeleton_endpoint_count",
        "skeleton_branchpoint_count",
        "thin_spur_score",
        "severe_qc_failure",
        "requires_review",
    ):
        qc_group[name][row_slice] = getattr(batch, name)


def write_subject_body_mask_qc_group(
    root: zarr.Group,
    *,
    refined_run: Optional[str] = None,
    chunk_size: int = 256,
    overwrite: bool = True,
    policy: SubjectBodyMaskQcPolicy | None = None,
) -> dict[str, object]:
    """Compute and persist ``components/subject_body/qc`` for a refined run."""

    resolved_policy = policy or SubjectBodyMaskQcPolicy()
    run_name, _cached_group = _resolve_refined_run(root, refined_run)
    refined_group = resolve_mutable_refined_subject_mask_run(root, run_name)
    body_idx = _subject_body_channel_index(refined_group)
    if not _available_channel(refined_group, body_idx):
        raise ValueError(f"refined_subject_masks_runs/{run_name} subject_body channel is unavailable.")

    try:
        mask_store = open_mask_store(
            refined_group,
            source_path=f"refined_subject_masks_runs/{run_name}",
            prefer="dense",
        )
    except (MaskStoreError, ValueError) as exc:
        raise ValueError(
            f"refined_subject_masks_runs/{run_name} missing usable mask store (masks_roi or mask_rle)."
        ) from exc
    if len(tuple(mask_store.shape)) != 4:
        raise ValueError(f"refined_subject_masks_runs/{run_name} mask store must be 4D, got {tuple(mask_store.shape)}.")
    row_count = int(mask_store.n_rows)
    chunk = max(1, int(chunk_size))
    qc_group = _prepare_qc_group(refined_group, row_count=row_count, policy=resolved_policy, overwrite=overwrite)
    reason_labels = np.full((row_count,), "ok", dtype=object)

    start_time = time.perf_counter()
    for start in range(0, row_count, chunk):
        stop = min(row_count, start + chunk)
        row_slice = slice(start, stop)
        batch_masks = np.asarray(mask_store.read_dense(rows=row_slice, channels=body_idx)[:, 0], dtype=np.uint8)
        batch = compute_subject_body_mask_qc(batch_masks, policy=resolved_policy)
        _write_qc_batch(qc_group, row_slice, batch)
        reason_labels[row_slice] = batch.reason_labels

    write_reason_columns(
        qc_group,
        reason_labels,
        chunk_size=refined_subject_mask_metric_row_chunk(row_count),
        overwrite=True,
    )
    updated_at = _utc_now()
    reason_counts = _reason_counts(reason_labels)
    severe_count = int(np.count_nonzero(qc_group["severe_qc_failure"][:]))
    requires_review_count = int(np.count_nonzero(qc_group["requires_review"][:]))
    duration_seconds = float(time.perf_counter() - start_time)
    qc_group.attrs.update(
        {
            "updated_at_utc": updated_at,
            "duration_seconds": duration_seconds,
            "row_count": row_count,
            "severe_qc_failure_count": severe_count,
            "requires_review_count": requires_review_count,
            "reason_counts": reason_counts,
        }
    )
    refined_group.attrs["subject_body_mask_qc_status"] = "computed"
    refined_group.attrs["subject_body_mask_qc_schema_id"] = SUBJECT_BODY_MASK_QC_SCHEMA_ID
    refined_group.attrs["subject_body_mask_qc_updated_at_utc"] = updated_at
    refined_group.attrs["subject_body_mask_qc_severe_count"] = severe_count
    refined_group.attrs["subject_body_mask_qc_requires_review_count"] = requires_review_count
    return {
        "status": "updated",
        "refined_run": run_name,
        "row_count": row_count,
        "chunk_size": chunk,
        "duration_seconds": duration_seconds,
        "severe_qc_failure_count": severe_count,
        "requires_review_count": requires_review_count,
        "reason_counts": reason_counts,
        "updated_at_utc": updated_at,
    }


def write_subject_body_mask_qc(
    zarr_path: str | Path,
    *,
    refined_run: Optional[str] = None,
    chunk_size: int = 256,
    overwrite: bool = True,
    policy: SubjectBodyMaskQcPolicy | None = None,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    return write_subject_body_mask_qc_group(
        root,
        refined_run=refined_run,
        chunk_size=chunk_size,
        overwrite=overwrite,
        policy=policy,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute refined subject_body mask QC under components/subject_body/qc."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette analysis zarr archive.")
    parser.add_argument("--refined-run", help="refined_subject_masks_runs/<run>; defaults to latest.")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true", default=True, help="Replace existing subject_body/qc.")
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite", help="Fail if subject_body/qc exists.")
    parser.add_argument("--min-area-px", type=float, default=SubjectBodyMaskQcPolicy.min_area_px)
    parser.add_argument("--max-area-px", type=float)
    parser.add_argument("--min-largest-component-fraction", type=float, default=SubjectBodyMaskQcPolicy.min_largest_component_fraction)
    parser.add_argument("--min-solidity", type=float, default=SubjectBodyMaskQcPolicy.min_solidity)
    parser.add_argument("--max-hole-area-fraction", type=float, default=SubjectBodyMaskQcPolicy.max_hole_area_fraction)
    parser.add_argument("--max-skeleton-endpoint-count", type=int, default=SubjectBodyMaskQcPolicy.max_skeleton_endpoint_count)
    parser.add_argument("--max-skeleton-branchpoint-count", type=int, default=SubjectBodyMaskQcPolicy.max_skeleton_branchpoint_count)
    parser.add_argument("--max-thin-spur-score", type=float, default=SubjectBodyMaskQcPolicy.max_thin_spur_score)
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    policy = SubjectBodyMaskQcPolicy(
        min_area_px=float(args.min_area_px),
        max_area_px=None if args.max_area_px is None else float(args.max_area_px),
        min_largest_component_fraction=float(args.min_largest_component_fraction),
        min_solidity=None if args.min_solidity is None else float(args.min_solidity),
        max_hole_area_fraction=float(args.max_hole_area_fraction),
        max_skeleton_endpoint_count=int(args.max_skeleton_endpoint_count),
        max_skeleton_branchpoint_count=int(args.max_skeleton_branchpoint_count),
        max_thin_spur_score=float(args.max_thin_spur_score),
    )
    summary = write_subject_body_mask_qc(
        args.zarr_path,
        refined_run=args.refined_run,
        chunk_size=int(args.chunk_size),
        overwrite=bool(args.overwrite),
        policy=policy,
    )
    print(json.dumps(_json_safe(summary), indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
