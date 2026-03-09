#!/usr/bin/env python3
"""Show raw/refined eye-mask provenance for a Palette zarr archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import zarr

RAW_ATTR_KEYS: Sequence[str] = (
    "method",
    "segmenter",
    "segmenter_label_mode",
    "source_eye_masks_run",
    "source_keypoints_run",
    "source_keypoint_run",
    "source_keypoint_group",
    "source_crop_run",
    "source_crop_storage_mode",
    "source_crop_signature",
    "source_crop_revision",
    "source_roi_read_mode",
    "source_roi_cache_used",
    "probabilities_dtype",
    "probabilities_encoding",
    "probabilities_channels",
    "probabilities_source",
    "mask_probability_threshold",
    "mask_probs_chunk_rois",
    "inference_batch_size",
    "inference_duration_seconds",
    "eye_labels",
    "timing_profile",
    "provenance",
)

REFINED_ATTR_KEYS: Sequence[str] = (
    "method",
    "source_eye_masks_run",
    "source_eye_masks_method",
    "source_keypoints_run",
    "source_keypoint_run",
    "source_keypoint_group",
    "source_crop_run",
    "ellipse_fit_backend",
    "ellipse_fit_method",
    "ellipse_contour_mode",
    "mask_probability_threshold",
    "mask_probability_threshold_source",
    "success_min_eye_area_px",
    "traditional_fast_path",
    "force_refine_traditional",
    "refined_probabilities_written",
    "mask_probability_source",
    "mask_binary_source",
    "mask_binary_identity",
    "mask_probability_identity",
    "eye_labels",
    "summary_statistics",
    "metrics_summary",
    "reason_tag_counts",
    "eye_mask_review_status",
    "provenance",
)

ARRAY_NAMES: Sequence[str] = (
    "masks_roi",
    "mask_probs_roi",
    "mask_probs_roi_refined",
    "ellipse_params",
    "ellipse_success",
    "frame_indices",
    "detection_indices",
    "frame_counts",
)


def _sorted_group_keys(group: Optional[zarr.Group]) -> list[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    return sorted(key for key in keys if isinstance(key, str))


def _pick_run_name(parent: Optional[zarr.Group], explicit: Optional[str]) -> Optional[str]:
    if parent is None:
        return None
    if explicit:
        return explicit if explicit in parent else None
    latest = parent.attrs.get("latest")
    if isinstance(latest, str) and latest in parent:
        return latest
    keys = _sorted_group_keys(parent)
    return keys[-1] if keys else None


def _select_attrs(run: zarr.Group, keys: Sequence[str]) -> dict[str, object]:
    attrs = dict(run.attrs)
    return {key: attrs.get(key) for key in keys if key in attrs}


def _array_summary(run: zarr.Group) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name in ARRAY_NAMES:
        arr = run.get(name)
        if arr is None:
            continue
        out[name] = {
            "shape": tuple(int(v) for v in arr.shape),
            "dtype": str(arr.dtype),
            "chunks": tuple(int(v) for v in arr.chunks) if getattr(arr, "chunks", None) else None,
        }
    return out


def show_eye_mask_provenance(
    zarr_path: Path,
    *,
    raw_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    show_arrays: bool = True,
    show_summary: bool = True,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)

    raw_parent = root.get("eye_masks_runs")
    refined_parent = root.get("refined_eye_masks_runs")

    raw_run_name = _pick_run_name(raw_parent, raw_run)
    refined_run_name = _pick_run_name(refined_parent, refined_run)

    if raw_parent is None and refined_parent is None:
        raise ValueError(f"No eye_masks_runs or refined_eye_masks_runs found in {zarr_path}.")

    if raw_parent is not None and raw_run is not None and raw_run_name is None:
        raise ValueError(f"Raw eye-mask run '{raw_run}' not found in {zarr_path}.")
    if refined_parent is not None and refined_run is not None and refined_run_name is None:
        raise ValueError(f"Refined eye-mask run '{refined_run}' not found in {zarr_path}.")

    if show_summary:
        print(f"zarr_path: {zarr_path}")
        print(f"raw_run: {raw_run_name or 'none'}")
        print(f"refined_run: {refined_run_name or 'none'}")
        print()

    if raw_run_name and raw_parent is not None:
        run = raw_parent[raw_run_name]
        print(f"=== eye_masks_runs/{raw_run_name} attrs ===")
        print(json.dumps(_select_attrs(run, RAW_ATTR_KEYS), indent=2, default=str, sort_keys=True))
        print()
        print(f"=== eye_masks_runs/{raw_run_name} recommended threshold ===")
        print(
            json.dumps(
                {
                    "recommended_probability_threshold": run.attrs.get("recommended_probability_threshold"),
                    "recommended_probability_threshold_review": run.attrs.get("recommended_probability_threshold_review"),
                },
                indent=2,
                default=str,
                sort_keys=True,
            )
        )
        print()
        if show_arrays:
            print(f"=== eye_masks_runs/{raw_run_name} arrays ===")
            print(json.dumps(_array_summary(run), indent=2, default=str, sort_keys=True))
            print()

    if refined_run_name and refined_parent is not None:
        run = refined_parent[refined_run_name]
        print(f"=== refined_eye_masks_runs/{refined_run_name} attrs ===")
        print(json.dumps(_select_attrs(run, REFINED_ATTR_KEYS), indent=2, default=str, sort_keys=True))
        print()
        metrics_summary = run.attrs.get("metrics_summary", {})
        reason_tag_counts = metrics_summary.get("reason_tag_counts", {}) if isinstance(metrics_summary, dict) else {}
        print(f"=== refined_eye_masks_runs/{refined_run_name} reason_tag_counts ===")
        print(json.dumps(reason_tag_counts, indent=2, default=str, sort_keys=True))
        print()
        if show_arrays:
            print(f"=== refined_eye_masks_runs/{refined_run_name} arrays ===")
            print(json.dumps(_array_summary(run), indent=2, default=str, sort_keys=True))
            print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show raw/refined eye-mask provenance for a Palette zarr archive.")
    parser.add_argument("zarr_path", type=Path, help="Path to the analysis zarr.")
    parser.add_argument("--raw-run", help="Raw eye-mask run under eye_masks_runs (default: latest).")
    parser.add_argument("--refined-run", help="Refined eye-mask run under refined_eye_masks_runs (default: latest).")
    parser.add_argument("--no-arrays", action="store_true", help="Do not print array shape/dtype/chunk summaries.")
    parser.add_argument("--no-summary", action="store_true", help="Do not print the top-level run selection summary.")
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    show_eye_mask_provenance(
        args.zarr_path,
        raw_run=args.raw_run,
        refined_run=args.refined_run,
        show_arrays=not args.no_arrays,
        show_summary=not args.no_summary,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
