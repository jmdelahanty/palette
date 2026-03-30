#!/usr/bin/env python3
"""Visual compare tool for raw union masks and refined left/right ellipse fits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.widgets import Button, CheckButtons, Slider

from ..diagnostics.review_refined_eye_mask_failures import (
    _reason_tag_indices,
    _resolve_source_crop_run,
    _resolve_source_eye_run,
    _resolve_source_keypoint_run,
    _select_indices,
)
from ..refinement.refine_eye_masks import _measure_mask
from ..shared.crop_image_source import CropImageSource
from ..shared.mask_source import load_mask_bundle
from .visualize_eye_mask_patches import _fit_ellipse_from_mask
from .visualize_eye_masks import EyeMaskViewer, normalize_roi, open_zarr


@dataclass
class EllipseFitResult:
    name: str
    success: bool
    params: np.ndarray
    contour_xy: Optional[np.ndarray]
    failure_reason: Optional[str]


def _should_draw_fit(name: str, component: str, visibility: dict[str, bool]) -> bool:
    return bool(visibility.get(f"{name}_{component}", False))


def _compute_fit_comparison(mask: np.ndarray) -> tuple[EllipseFitResult, EllipseFitResult]:
    cv2_params, cv2_success, cv2_contour, _centroid = _fit_ellipse_from_mask(mask)
    sk_success, sk_params, _sk_centroid, sk_contour, sk_failure = _measure_mask(mask.astype(np.uint8))
    return (
        EllipseFitResult(
            name="cv2",
            success=bool(cv2_success),
            params=np.asarray(cv2_params, dtype=np.float32),
            contour_xy=None if cv2_contour is None else np.asarray(cv2_contour, dtype=np.float32),
            failure_reason=None if cv2_success else "fit_failed",
        ),
        EllipseFitResult(
            name="skimage",
            success=bool(sk_success),
            params=np.asarray(sk_params, dtype=np.float32),
            contour_xy=None if sk_contour is None else np.asarray(sk_contour, dtype=np.float32),
            failure_reason=sk_failure,
        ),
    )


def _ellipse_curve(params: np.ndarray, *, num_points: int = 100) -> np.ndarray:
    if params is None or len(params) < 5:
        return np.zeros((0, 2), dtype=np.float32)
    cx, cy, major, minor, theta = [float(v) for v in params[:5]]
    return EyeMaskViewer._ellipse_curve(cx, cy, major, minor, theta, num_points=num_points)


def _variant_mask_bundle(root: zarr.Group, group_path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    group = root[group_path]
    threshold_attr = group.attrs.get("mask_probability_threshold", 0.5)
    try:
        threshold = float(threshold_attr)
    except (TypeError, ValueError):
        threshold = 0.5
    bundle = load_mask_bundle(group, threshold=threshold, prefer_probs=True, materialize=False, lazy=True)
    return bundle.binary, bundle.probs


def create_fit_comparison_viewer(
    zarr_path: Path,
    *,
    refined_run: str,
    reason_tag: str = "ellipse_fail_pair",
    eye_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    keypoint_run: Optional[str] = None,
    limit: Optional[int] = 200,
    seed: int = 0,
) -> None:
    root = open_zarr(zarr_path)
    if "refined_eye_masks_runs" not in root or refined_run not in root["refined_eye_masks_runs"]:
        raise ValueError(f"Refined run '{refined_run}' not found in {zarr_path}.")

    eye_run_name = _resolve_source_eye_run(root, refined_run, eye_run)
    crop_run_name = _resolve_source_crop_run(root, refined_run, eye_run_name, crop_run)
    keypoint_run_name = _resolve_source_keypoint_run(root, refined_run, eye_run_name, keypoint_run)

    reason_ds = root[f"refined_eye_masks_runs/{refined_run}/metrics/reason"]
    reason_values = np.asarray(reason_ds[:], dtype=object)
    matched_indices = _reason_tag_indices(reason_values, reason_tag)
    if not matched_indices:
        raise ValueError(
            f"No ROIs in refined run '{refined_run}' matched reason tag '{reason_tag}'."
        )
    selected_indices = _select_indices(matched_indices, limit, seed)

    crop_source = CropImageSource.open(root, crop_run=crop_run_name, zarr_path=zarr_path)
    kp_group = root[f"keypoints_runs/{keypoint_run_name}"]
    keypoints = np.asarray(kp_group["keypoints_roi"][:], dtype=np.float32)
    keypoint_labels = list(kp_group.attrs.get("keypoint_labels", []))
    raw_masks, raw_probs = _variant_mask_bundle(root, f"eye_masks_runs/{eye_run_name}")
    refined_masks, _refined_probs = _variant_mask_bundle(root, f"refined_eye_masks_runs/{refined_run}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax_raw, ax_left, ax_right = axes
    plt.subplots_adjust(bottom=0.2)

    slider_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04])
    slider = Slider(
        ax=slider_ax,
        label="Subset ROI",
        valmin=0,
        valmax=len(selected_indices) - 1,
        valinit=0,
        valfmt="%0.0f",
    )

    ax_prev = fig.add_axes([0.02, 0.08, 0.1, 0.05])
    ax_next = fig.add_axes([0.88, 0.08, 0.1, 0.05])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")
    checkbox_ax = fig.add_axes([0.02, 0.72, 0.18, 0.22])
    fit_visibility = {
        "cv2_contour": True,
        "cv2_ellipse": True,
        "skimage_contour": True,
        "skimage_ellipse": True,
    }
    check = CheckButtons(
        checkbox_ax,
        labels=["cv2 contour", "cv2 ellipse", "skimage contour", "skimage ellipse"],
        actives=[
            fit_visibility["cv2_contour"],
            fit_visibility["cv2_ellipse"],
            fit_visibility["skimage_contour"],
            fit_visibility["skimage_ellipse"],
        ],
    )
    checkbox_ax.set_title("Show Overlays", fontsize=9)

    def _draw_eye_panel(ax, roi: np.ndarray, eye_mask: np.ndarray, label: str) -> None:
        ax.clear()
        ax.imshow(roi, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.imshow(np.ma.masked_where(eye_mask <= 0, eye_mask), cmap="Blues", alpha=0.45, interpolation="nearest")

        cv2_fit, sk_fit = _compute_fit_comparison(eye_mask)
        if (
            _should_draw_fit("cv2", "contour", fit_visibility)
            and cv2_fit.contour_xy is not None
            and cv2_fit.contour_xy.size > 0
        ):
            ax.plot(cv2_fit.contour_xy[:, 0], cv2_fit.contour_xy[:, 1], color="#00d1b2", linewidth=0.8, alpha=0.8)
        if (
            _should_draw_fit("skimage", "contour", fit_visibility)
            and sk_fit.contour_xy is not None
            and sk_fit.contour_xy.size > 0
        ):
            ax.plot(sk_fit.contour_xy[:, 0], sk_fit.contour_xy[:, 1], color="#ffd166", linewidth=0.8, alpha=0.8)

        cv2_curve = _ellipse_curve(cv2_fit.params) if cv2_fit.success else np.zeros((0, 2), dtype=np.float32)
        sk_curve = _ellipse_curve(sk_fit.params) if sk_fit.success else np.zeros((0, 2), dtype=np.float32)
        if _should_draw_fit("cv2", "ellipse", fit_visibility) and cv2_curve.size > 0:
            ax.plot(cv2_curve[:, 0], cv2_curve[:, 1], color="#00d1b2", linewidth=1.4, label="cv2")
        if _should_draw_fit("skimage", "ellipse", fit_visibility) and sk_curve.size > 0:
            ax.plot(sk_curve[:, 0], sk_curve[:, 1], color="#ff5d8f", linewidth=1.2, linestyle="--", label="skimage")

        ax.set_axis_off()
        ax.set_title(
            "{label}\ncv2={cv2} | sk={sk}".format(
                label=label,
                cv2="ok" if cv2_fit.success else "fail",
                sk="ok" if sk_fit.success else (sk_fit.failure_reason or "fail"),
            ),
            fontsize=10,
        )

    def _update(local_idx: int) -> None:
        global_idx = int(selected_indices[local_idx])
        roi = np.asarray(crop_source.read_slice(global_idx, global_idx + 1)[0])
        roi_norm = normalize_roi(roi)
        raw_union = np.asarray(raw_masks[global_idx], dtype=np.uint8)
        if raw_union.ndim == 3 and raw_union.shape[0] > 0:
            raw_union = raw_union[0]
        refined_row = np.asarray(refined_masks[global_idx], dtype=np.uint8)
        kp_row = np.asarray(keypoints[global_idx], dtype=np.float32)
        kp_valid = np.all(np.isfinite(kp_row), axis=1)
        reason = str(reason_values[global_idx])

        ax_raw.clear()
        ax_raw.imshow(roi_norm, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_raw.imshow(np.ma.masked_where(raw_union <= 0, raw_union), cmap="magma", alpha=0.4, interpolation="nearest")
        for kp_idx, is_valid in enumerate(kp_valid):
            if not is_valid:
                continue
            x, y = kp_row[kp_idx]
            ax_raw.plot([x], [y], marker="o", markersize=5, markeredgecolor="black", markerfacecolor=EyeMaskViewer._keypoint_color(keypoint_labels[kp_idx] if kp_idx < len(keypoint_labels) else f"kp_{kp_idx}"))
        prob_note = ""
        if raw_probs is not None:
            probs_row = np.asarray(raw_probs[global_idx], dtype=np.float32)
            prob_note = f" | probs={tuple(probs_row.shape)}"
        ax_raw.set_title(
            "ROI {local}/{total} (global {global_idx})\n{reason}{prob_note}".format(
                local=local_idx + 1,
                total=len(selected_indices),
                global_idx=global_idx + 1,
                reason=reason,
                prob_note=prob_note,
            ),
            fontsize=10,
        )
        ax_raw.set_axis_off()

        _draw_eye_panel(ax_left, roi_norm, refined_row[0], "Left refined mask")
        _draw_eye_panel(ax_right, roi_norm, refined_row[1], "Right refined mask")
        fig.canvas.draw_idle()

    def _step(delta: int) -> None:
        idx = int(round(slider.val)) + delta
        idx = max(0, min(len(selected_indices) - 1, idx))
        slider.set_val(idx)

    slider.on_changed(lambda val: _update(int(round(val))))
    btn_prev.on_clicked(lambda _evt: _step(-1))
    btn_next.on_clicked(lambda _evt: _step(1))
    label_to_key = {
        "cv2 contour": "cv2_contour",
        "cv2 ellipse": "cv2_ellipse",
        "skimage contour": "skimage_contour",
        "skimage ellipse": "skimage_ellipse",
    }

    def _toggle_overlay(label: str) -> None:
        key = label_to_key.get(str(label))
        if key is None:
            return
        fit_visibility[key] = not fit_visibility.get(key, False)
        _update(int(round(slider.val)))

    check.on_clicked(_toggle_overlay)

    _update(0)
    fig.canvas.manager.set_window_title(
        f"Ellipse Fit Comparison | refined={refined_run} reason_tag={reason_tag}"
    )
    try:
        plt.show()
    finally:
        crop_source.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visually compare cv2 and skimage ellipse fitters on refined eye-mask failures."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the analysis zarr.")
    parser.add_argument("--refined-run", required=True, help="Refined eye-mask run name.")
    parser.add_argument(
        "--reason-tag",
        default="ellipse_fail_pair",
        help="Reason tag used to select ROIs (default: ellipse_fail_pair).",
    )
    parser.add_argument("--eye-run", help="Explicit source eye-mask run override.")
    parser.add_argument("--crop-run", help="Explicit crop run override.")
    parser.add_argument("--keypoint-run", help="Explicit keypoint run override.")
    parser.add_argument("--limit", type=int, default=200, help="Max ROIs to view.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for ROI subsampling.")
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    create_fit_comparison_viewer(
        args.zarr_path,
        refined_run=args.refined_run,
        reason_tag=args.reason_tag,
        eye_run=args.eye_run,
        crop_run=args.crop_run,
        keypoint_run=args.keypoint_run,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
