#!/usr/bin/env python3
"""Open the eye-mask viewer on a subset of refined failure-tag ROIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import zarr

from ..shared.provenance_attrs import resolve_source_keypoints_run
from ..visualization.visualize_eye_masks import create_viewer


def _resolve_source_eye_run(
    root: zarr.Group, refined_run: str, explicit_source_run: Optional[str]
) -> str:
    if explicit_source_run:
        return explicit_source_run
    refined = root["refined_eye_masks_runs"][refined_run]
    source_run = refined.attrs.get("source_eye_masks_run")
    if not source_run:
        raise ValueError(
            f"Refined run '{refined_run}' is missing source_eye_masks_run. Pass --eye-run explicitly."
        )
    return str(source_run)


def _resolve_source_crop_run(
    root: zarr.Group,
    refined_run: str,
    eye_run: str,
    explicit_crop_run: Optional[str],
) -> str:
    if explicit_crop_run:
        return explicit_crop_run
    refined = root["refined_eye_masks_runs"][refined_run]
    source_run = root["eye_masks_runs"][eye_run]
    for group in (refined, source_run):
        crop_run = group.attrs.get("source_crop_run")
        if crop_run:
            return str(crop_run)
    raise ValueError(
        f"Unable to resolve source_crop_run for refined run '{refined_run}'. Pass --crop-run explicitly."
    )


def _resolve_source_keypoint_run(
    root: zarr.Group,
    refined_run: str,
    eye_run: str,
    explicit_keypoint_run: Optional[str],
) -> str:
    if explicit_keypoint_run:
        return explicit_keypoint_run
    refined = root["refined_eye_masks_runs"][refined_run]
    eye_group = root["eye_masks_runs"][eye_run]
    keypoint_run = resolve_source_keypoints_run(refined.attrs)
    if keypoint_run:
        return str(keypoint_run)
    keypoint_run = resolve_source_keypoints_run(eye_group.attrs)
    if keypoint_run:
        return str(keypoint_run)
    raise ValueError(
        f"Unable to resolve source keypoint run for refined run '{refined_run}'. Pass --keypoint-run explicitly."
    )


def _reason_tag_indices(reason_values: Sequence[object], tag: str) -> List[int]:
    matched: List[int] = []
    for idx, value in enumerate(reason_values):
        parts = [part for part in str(value).split("|") if part]
        if tag in parts:
            matched.append(idx)
    return matched


def _select_indices(indices: Sequence[int], limit: Optional[int], seed: int) -> List[int]:
    selected = list(indices)
    if limit is None or limit <= 0 or len(selected) <= limit:
        return selected
    rng = np.random.default_rng(seed)
    chosen = sorted(int(val) for val in rng.choice(selected, size=limit, replace=False))
    return chosen


def review_refined_eye_mask_failures(
    zarr_path: Path,
    *,
    refined_run: str,
    reason_tag: str,
    eye_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    keypoint_run: Optional[str] = None,
    limit: Optional[int] = 200,
    seed: int = 0,
    frame_flag_file: str = "eye_mask_frame_flags.json",
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    if "refined_eye_masks_runs" not in root:
        raise ValueError(f"No refined_eye_masks_runs group found in {zarr_path}.")
    if refined_run not in root["refined_eye_masks_runs"]:
        raise ValueError(
            f"Refined run '{refined_run}' not found under refined_eye_masks_runs in {zarr_path}."
        )

    eye_run_name = _resolve_source_eye_run(root, refined_run, eye_run)
    crop_run_name = _resolve_source_crop_run(root, refined_run, eye_run_name, crop_run)
    keypoint_run_name = _resolve_source_keypoint_run(
        root, refined_run, eye_run_name, keypoint_run
    )

    reason_ds = root[f"refined_eye_masks_runs/{refined_run}/metrics/reason"]
    reason_values = np.asarray(reason_ds[:], dtype=object)
    matched_indices = _reason_tag_indices(reason_values, reason_tag)
    if not matched_indices:
        raise ValueError(
            f"No ROIs in refined run '{refined_run}' matched reason tag '{reason_tag}'."
        )
    selected_indices = _select_indices(matched_indices, limit, seed)

    print(
        "Opening {count} ROI(s) with tag '{tag}' from {run} "
        "(matched={matched}, eye_run={eye}, crop_run={crop}, keypoint_run={kp})".format(
            count=len(selected_indices),
            tag=reason_tag,
            run=refined_run,
            matched=len(matched_indices),
            eye=eye_run_name,
            crop=crop_run_name,
            kp=keypoint_run_name,
        )
    )

    create_viewer(
        zarr_path,
        eye_run_name,
        crop_run_name,
        keypoint_run_name,
        refined_runs=[refined_run],
        frame_flag_file=frame_flag_file,
        roi_indices=selected_indices,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review a refined eye-mask run on ROIs matching a given reason tag."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the analysis zarr.")
    parser.add_argument(
        "--refined-run",
        required=True,
        help="Refined eye-mask run name under refined_eye_masks_runs.",
    )
    parser.add_argument(
        "--reason-tag",
        default="ellipse_fail_pair",
        help="Reason tag to filter on (default: ellipse_fail_pair).",
    )
    parser.add_argument("--eye-run", help="Explicit source eye mask run override.")
    parser.add_argument("--crop-run", help="Explicit crop run override.")
    parser.add_argument("--keypoint-run", help="Explicit keypoint run override.")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of matching ROIs to review (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when subsampling matches (default: 0).",
    )
    parser.add_argument(
        "--frame-flag-file",
        default="eye_mask_frame_flags.json",
        help="JSON file to append cleanup flags when pressing 'b'.",
    )
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    review_refined_eye_mask_failures(
        args.zarr_path,
        refined_run=args.refined_run,
        reason_tag=args.reason_tag,
        eye_run=args.eye_run,
        crop_run=args.crop_run,
        keypoint_run=args.keypoint_run,
        limit=args.limit,
        seed=args.seed,
        frame_flag_file=args.frame_flag_file,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
