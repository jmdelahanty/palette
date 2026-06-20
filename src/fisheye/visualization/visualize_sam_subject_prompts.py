"""Visualize SAM subject-mask prompts on Palette ROI crops."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import zarr

from fisheye.shared.mask_store import MaskStore, MaskStoreError, open_mask_store
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.run_sam_subject_masks import (
    BOX_PROMPT_SOURCE_CHOICES,
    DEFAULT_BOX_PROMPT_SOURCE,
    DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
    DEFAULT_NEGATIVE_POINT_POLICY,
    DEFAULT_ROI_INSET_FRACTION,
    build_roi_inset_box_xyxy,
    build_point_prompt_coords_labels,
    NEGATIVE_POINT_POLICY_CHOICES,
    KEYPOINT_GROUP_CHOICES,
    compute_row_eligibility,
    project_bbox_norm_to_roi_xyxy,
    resolve_prompt_keypoint_selection,
    resolve_sam_subject_inputs,
)
from fisheye.utils.zarr_io import open_zarr_root

WINDOW_NAME = "SAM Subject Prompt Visualizer"
RESULT_WINDOW_NAME = "SAM Subject Prompt Result Overlay"
DISPLAY_SCALE = 2.0
POINT_COLOR = (0, 0, 255)
NEGATIVE_POINT_COLOR = (255, 128, 0)
BOX_COLOR = (0, 255, 255)
MASK_COLOR = (0, 220, 0)


@dataclass(frozen=True)
class LoadedSubjectRun:
    run_name: str
    mask_labels: tuple[str, ...]
    available_channels: np.ndarray
    masks_roi: Any | None
    mask_store: MaskStore | None = None


def _require_gui_display() -> None:
    display = str(os.environ.get("DISPLAY") or "").strip()
    wayland_display = str(os.environ.get("WAYLAND_DISPLAY") or "").strip()
    if display or wayland_display:
        return

    message = (
        "No GUI display is available for the OpenCV visualizer window. "
        "DISPLAY and WAYLAND_DISPLAY are unset in this shell."
    )
    if os.environ.get("TMUX"):
        message += (
            " This shell is inside tmux, so the pane likely missed the GUI environment. "
            "Open a new tmux window from a GUI-capable session or export DISPLAY/XAUTHORITY first."
        )
    raise RuntimeError(message)


def _load_subject_run(root: zarr.Group, subject_run: str | None) -> LoadedSubjectRun | None:
    if not subject_run:
        return None
    run_group, run_name = resolve_zarr_run(
        root,
        "subject_mask_runs",
        subject_run,
        fallback_to_latest=True,
        run_label="Subject-mask run",
    )
    labels_raw = run_group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RuntimeError(f"subject_mask_runs/{run_name} missing usable mask_labels attr.")
    available = run_group.get("available_channels")
    if available is None:
        raise RuntimeError(f"subject_mask_runs/{run_name} missing available_channels.")
    masks_roi = run_group.get("masks_roi")
    try:
        mask_store = open_mask_store(
            run_group,
            source_path=f"subject_mask_runs/{run_name}",
            prefer="dense",
        )
    except MaskStoreError as exc:
        raise RuntimeError(f"subject_mask_runs/{run_name} missing usable mask store (masks_roi or mask_rle).") from exc
    return LoadedSubjectRun(
        run_name=run_name,
        mask_labels=tuple(str(item) for item in labels_raw),
        available_channels=np.asarray(available[:], dtype=bool),
        masks_roi=masks_roi,
        mask_store=mask_store,
    )


def _subject_body_mask_for_row(subject_run: LoadedSubjectRun | None, row_idx: int) -> np.ndarray | None:
    if subject_run is None:
        return None
    if "subject_body" not in subject_run.mask_labels:
        return None
    channel_idx = subject_run.mask_labels.index("subject_body")
    if channel_idx >= int(subject_run.available_channels.shape[0]) or not bool(subject_run.available_channels[channel_idx]):
        return None
    if subject_run.mask_store is not None:
        return np.asarray(subject_run.mask_store.read_dense(rows=int(row_idx), channels=channel_idx)[0, 0], dtype=np.uint8)
    if subject_run.masks_roi is None:
        return None
    return np.asarray(subject_run.masks_roi[int(row_idx), channel_idx], dtype=np.uint8)


def _to_bgr(roi_image: np.ndarray) -> np.ndarray:
    image = np.asarray(roi_image, dtype=np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and int(image.shape[2]) == 3:
        return image.copy()
    if image.ndim == 3 and int(image.shape[2]) == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported ROI image shape for visualization: {tuple(int(v) for v in image.shape)}")


def draw_prompt_overlay(
    roi_image: np.ndarray,
    *,
    point_xy: np.ndarray | None = None,
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    box_xyxy: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    title: str,
    footer_lines: Sequence[str] = (),
) -> np.ndarray:
    canvas = _to_bgr(roi_image)
    overlay = canvas.copy()

    if mask is not None:
        overlay[np.asarray(mask, dtype=np.uint8) > 0] = MASK_COLOR
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

    if box_xyxy is not None:
        x0, y0, x1, y1 = [int(round(float(v))) for v in np.asarray(box_xyxy, dtype=np.float32).reshape(4)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), BOX_COLOR, 1)

    if point_coords is not None:
        coords = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
        labels = (
            np.asarray(point_labels, dtype=np.int32).reshape(-1)
            if point_labels is not None
            else np.ones((coords.shape[0],), dtype=np.int32)
        )
        for coord, label in zip(coords, labels):
            px, py = [int(round(float(v))) for v in coord.tolist()]
            color = POINT_COLOR if int(label) == 1 else NEGATIVE_POINT_COLOR
            cv2.circle(canvas, (px, py), 3, color, -1)
            cv2.circle(canvas, (px, py), 7, color, 1)
    elif point_xy is not None:
        px, py = [int(round(float(v))) for v in np.asarray(point_xy, dtype=np.float32).reshape(2)]
        cv2.circle(canvas, (px, py), 3, POINT_COLOR, -1)
        cv2.circle(canvas, (px, py), 7, POINT_COLOR, 1)

    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    line_y = int(canvas.shape[0]) - 16 - (18 * max(0, len(footer_lines) - 1))
    for line in footer_lines:
        cv2.putText(
            canvas,
            str(line),
            (10, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        line_y += 18
    return canvas


def _eligibility_lines(eligibility: Any, row_idx: int) -> list[str]:
    lines = [
        f"eligible={bool(eligibility.eligible[int(row_idx)])}",
        f"finite={bool(eligibility.prompt_point_finite[int(row_idx)])}",
        f"in_bounds={bool(eligibility.prompt_point_in_bounds[int(row_idx)])}",
        f"prompt_count={int(eligibility.prompt_point_count[int(row_idx)])}",
        f"success_ok={bool(eligibility.success_ok[int(row_idx)])}",
        f"geometry_ok={bool(eligibility.geometry_ok[int(row_idx)])}",
        f"usable_ok={bool(eligibility.usable_ok[int(row_idx)])}",
        f"interpolated_skipped={bool(eligibility.skipped_interpolated[int(row_idx)])}",
    ]
    return lines


def _compose_prompt_grid(
    roi_image: np.ndarray,
    *,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    box_xyxy: np.ndarray | None,
    footer_lines: Sequence[str],
) -> np.ndarray:
    crop_panel = draw_prompt_overlay(roi_image, title="Crop ROI", footer_lines=footer_lines)
    point_panel = draw_prompt_overlay(
        roi_image,
        point_coords=point_coords,
        point_labels=point_labels,
        title="Point Prompt",
        footer_lines=(
            "positive=%d negative=%d"
            % (
                int(np.sum(np.asarray(point_labels, dtype=np.int32) == 1)),
                int(np.sum(np.asarray(point_labels, dtype=np.int32) == 0)),
            ),
        ),
    )
    if box_xyxy is None:
        box_panel = draw_prompt_overlay(
            roi_image,
            title="Box Prompt",
            footer_lines=("disabled",),
        )
        combined_panel = draw_prompt_overlay(
            roi_image,
            point_coords=point_coords,
            point_labels=point_labels,
            title="Point + Box Prompt",
            footer_lines=("box disabled",),
        )
    else:
        box_panel = draw_prompt_overlay(
            roi_image,
            box_xyxy=box_xyxy,
            title="Box Prompt",
            footer_lines=(
                "box=[%.1f, %.1f, %.1f, %.1f]"
                % tuple(float(v) for v in np.asarray(box_xyxy, dtype=np.float32).reshape(4)),
            ),
        )
        combined_panel = draw_prompt_overlay(
            roi_image,
            point_coords=point_coords,
            point_labels=point_labels,
            box_xyxy=box_xyxy,
            title="Point + Box Prompt",
        )

    top = np.hstack([crop_panel, point_panel])
    bottom = np.hstack([box_panel, combined_panel])
    return np.vstack([top, bottom])


def launch_visualizer(
    zarr_path: str | Path,
    *,
    crop_run: str | None = None,
    keypoint_run: str | None = None,
    keypoint_group: str = "auto",
    subject_run: str | None = None,
    roi_index: int = 0,
    skip_interpolated: bool = True,
    use_box_prompt: bool = True,
    box_prompt_source: str = DEFAULT_BOX_PROMPT_SOURCE,
    roi_inset_fraction: float = DEFAULT_ROI_INSET_FRACTION,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
    negative_point_margin_fraction: float = DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
    positive_keypoint_labels: Sequence[str] | None = None,
) -> None:
    root = open_zarr_root(zarr_path, mode="r")
    inputs = resolve_sam_subject_inputs(
        root,
        crop_run=crop_run,
        keypoint_run=keypoint_run,
        keypoint_group=keypoint_group,
    )
    prompt_selection = resolve_prompt_keypoint_selection(
        inputs,
        positive_keypoint_labels=positive_keypoint_labels,
    )
    eligibility = compute_row_eligibility(
        inputs,
        prompt_selection=prompt_selection,
        skip_interpolated=skip_interpolated,
    )
    subject = _load_subject_run(root, subject_run)

    _require_gui_display()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1900, 1200)
    if subject is not None:
        cv2.namedWindow(RESULT_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(RESULT_WINDOW_NAME, 1100, 700)

    total_rows = int(inputs.row_count)
    current_idx = max(0, min(int(roi_index), total_rows - 1))

    def update_display() -> None:
        roi_image = np.asarray(inputs.roi_images[current_idx], dtype=np.uint8)
        point_xy = np.asarray(
            inputs.keypoints_roi[current_idx, prompt_selection.indices, :],
            dtype=np.float32,
        )
        finite = np.isfinite(point_xy).all(axis=1)
        in_bounds = finite.copy()
        in_bounds &= point_xy[:, 0] >= 0.0
        in_bounds &= point_xy[:, 1] >= 0.0
        in_bounds &= point_xy[:, 0] < float(inputs.roi_width)
        in_bounds &= point_xy[:, 1] < float(inputs.roi_height)
        if int(np.sum(in_bounds)) > 0:
            point_coords, point_labels = build_point_prompt_coords_labels(
                point_xy[in_bounds],
                roi_height=int(inputs.roi_height),
                roi_width=int(inputs.roi_width),
                negative_point_policy=negative_point_policy,
                negative_point_margin_fraction=negative_point_margin_fraction,
            )
        else:
            point_coords = np.zeros((0, 2), dtype=np.float32)
            point_labels = np.zeros((0,), dtype=np.int32)
        if use_box_prompt and box_prompt_source == "roi_inset":
            box_xyxy = build_roi_inset_box_xyxy(
                roi_height=int(inputs.roi_height),
                roi_width=int(inputs.roi_width),
                inset_fraction=float(roi_inset_fraction),
            )
        elif use_box_prompt:
            box_xyxy = project_bbox_norm_to_roi_xyxy(
                inputs.bbox_norm_coords[current_idx],
                inputs.roi_coordinates_full[current_idx],
                frame_height=int(inputs.frame_height),
                frame_width=int(inputs.frame_width),
                roi_height=int(inputs.roi_height),
                roi_width=int(inputs.roi_width),
            )
        else:
            box_xyxy = None
        footer_lines = [
            f"ROI {current_idx + 1}/{total_rows} frame={int(inputs.frame_indices[current_idx])} det={int(inputs.detection_indices[current_idx])}",
            *_eligibility_lines(eligibility, current_idx),
        ]
        grid = _compose_prompt_grid(
            roi_image,
            point_coords=point_coords,
            point_labels=point_labels,
            box_xyxy=box_xyxy,
            footer_lines=footer_lines,
        )
        if DISPLAY_SCALE != 1.0:
            grid = cv2.resize(grid, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW_NAME, grid)

        if subject is not None:
            mask = _subject_body_mask_for_row(subject, current_idx)
            result_panel = draw_prompt_overlay(
                roi_image,
                point_coords=point_coords,
                point_labels=point_labels,
                box_xyxy=box_xyxy,
                mask=mask,
                title=f"Stored subject_body ({subject.run_name})",
                footer_lines=(
                    "mask_present=%s"
                    % ("yes" if mask is not None and int(np.count_nonzero(mask)) > 0 else "no"),
                ),
            )
            if DISPLAY_SCALE != 1.0:
                result_panel = cv2.resize(
                    result_panel,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(RESULT_WINDOW_NAME, result_panel)

    print("\nSAM Subject Prompt Visualizer")
    print(f"  Zarr: {Path(zarr_path).expanduser().resolve()}")
    print(f"  Crop run: {inputs.crop_run}")
    print(f"  Keypoint run: {inputs.keypoint_group}/{inputs.keypoint_run}")
    print(f"  Keypoint labels: {list(inputs.keypoint_labels)}")
    print(f"  Positive keypoint labels: {list(prompt_selection.labels)}")
    print(f"  Rows: {inputs.row_count}")
    print(f"  Box prompt: {'enabled' if use_box_prompt else 'disabled'}")
    if use_box_prompt:
        print(f"  Box prompt source: {box_prompt_source}")
        if box_prompt_source == "roi_inset":
            print(f"  ROI inset fraction: {float(roi_inset_fraction):.3f}")
    print(f"  Negative point policy: {negative_point_policy}")
    if negative_point_policy != "none":
        print(f"  Negative point margin fraction: {float(negative_point_margin_fraction):.3f}")
    if subject is not None:
        print(f"  Subject run: subject_mask_runs/{subject.run_name}")
    print("Controls:")
    print("  n/p: next/previous ROI")
    print("  j/k: -/+10 ROI")
    print("  q/ESC: quit")

    update_display()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n") and current_idx < total_rows - 1:
            current_idx += 1
            update_display()
        elif key == ord("p") and current_idx > 0:
            current_idx -= 1
            update_display()
        elif key == ord("j") and current_idx > 0:
            current_idx = max(0, current_idx - 10)
            update_display()
        elif key == ord("k") and current_idx < total_rows - 1:
            current_idx = min(total_rows - 1, current_idx + 10)
            update_display()

    cv2.destroyAllWindows()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette training or analysis zarr to inspect.")
    parser.add_argument("--crop-run", type=str, help="Materialized crop run to read (default: latest materialized).")
    parser.add_argument("--keypoint-run", type=str, help="Keypoint run (default: latest refined, else latest raw).")
    parser.add_argument(
        "--keypoint-group",
        choices=list(KEYPOINT_GROUP_CHOICES),
        default="auto",
        help="Keypoint parent group preference (default: auto).",
    )
    parser.add_argument("--subject-run", type=str, help="Optional subject_mask_runs/<run> to overlay.")
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index.")
    parser.add_argument(
        "--include-interpolated",
        action="store_true",
        help="Do not mark interpolated rows as skipped in the eligibility footer.",
    )
    parser.add_argument(
        "--no-box-prompt",
        action="store_true",
        help="Disable box visualization to compare against point-only prompting.",
    )
    parser.add_argument(
        "--box-prompt-source",
        choices=list(BOX_PROMPT_SOURCE_CHOICES),
        default=DEFAULT_BOX_PROMPT_SOURCE,
        help=f"Box prompt source when enabled (default: {DEFAULT_BOX_PROMPT_SOURCE}).",
    )
    parser.add_argument(
        "--roi-inset-fraction",
        type=float,
        default=DEFAULT_ROI_INSET_FRACTION,
        help=(
            "Inset fraction for --box-prompt-source roi_inset, as a fraction of ROI width/height "
            f"(default: {DEFAULT_ROI_INSET_FRACTION})."
        ),
    )
    parser.add_argument(
        "--negative-point-policy",
        choices=list(NEGATIVE_POINT_POLICY_CHOICES),
        default=DEFAULT_NEGATIVE_POINT_POLICY,
        help=f"Add fixed background negative points to the visualized prompt (default: {DEFAULT_NEGATIVE_POINT_POLICY}).",
    )
    parser.add_argument(
        "--negative-point-margin-fraction",
        type=float,
        default=DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
        help=(
            "Border inset fraction for negative points, as a fraction of ROI width/height "
            f"(default: {DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION})."
        ),
    )
    parser.add_argument(
        "--positive-keypoint-labels",
        nargs="+",
        help=(
            "Optional keypoint labels to use as positive prompts. "
            "Defaults to all labels available on the resolved keypoint run."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    launch_visualizer(
        args.zarr_path,
        crop_run=args.crop_run,
        keypoint_run=args.keypoint_run,
        keypoint_group=args.keypoint_group,
        subject_run=args.subject_run,
        roi_index=int(args.roi_index),
        skip_interpolated=not args.include_interpolated,
        use_box_prompt=not args.no_box_prompt,
        box_prompt_source=args.box_prompt_source,
        roi_inset_fraction=float(args.roi_inset_fraction),
        negative_point_policy=args.negative_point_policy,
        negative_point_margin_fraction=float(args.negative_point_margin_fraction),
        positive_keypoint_labels=args.positive_keypoint_labels,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
