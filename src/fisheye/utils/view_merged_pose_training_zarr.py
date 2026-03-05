#!/usr/bin/env python3
"""Interactive viewer for merged pose-training Zarr datasets.

Supports:
- split-aware browsing (all/train/val/test)
- ROI overlay rendering (bbox + keypoints)
- optional export of all selected samples as PNG files
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import zarr
from matplotlib.widgets import Button, Slider


def _resolve_run(parent: zarr.Group, explicit: Optional[str]) -> str:
    if explicit:
        name = str(explicit).strip()
        if name in parent:
            return name
        raise ValueError(f"Run '{name}' not found. Available: {', '.join(list(parent.keys()))}")
    latest = parent.attrs.get("latest")
    if not latest or str(latest) not in parent:
        raise ValueError("Missing or invalid latest run attribute.")
    return str(latest)


def _load_selected_indices(root: zarr.Group, split: str) -> np.ndarray:
    crop_parent = root["crop_runs"]
    crop_run = _resolve_run(crop_parent, None)
    total_samples = int(root[f"crop_runs/{crop_run}/roi_images"].shape[0])

    if split == "all":
        return np.arange(total_samples, dtype=np.int64)

    split_name = f"{split}_indices"
    path = f"splits/{split_name}"
    if path not in root:
        raise ValueError(f"Split array missing: {path}")
    values = np.asarray(root[path][:], dtype=np.int64)
    return values


def _xywhn_to_rect(xywh: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, w, h = [float(v) for v in xywh.tolist()]
    x1 = (cx - (w / 2.0)) * float(width)
    y1 = (cy - (h / 2.0)) * float(height)
    return x1, y1, w * float(width), h * float(height)


def _draw_sample(
    *,
    ax,
    roi_image: np.ndarray,
    bbox_xywhn_pose: Optional[np.ndarray],
    keypoints_xy: np.ndarray,
    show_bbox: bool,
    show_keypoints: bool,
    keypoint_labels: list[str],
    title: str,
) -> None:
    ax.clear()
    if roi_image.ndim == 2:
        ax.imshow(roi_image, cmap="gray")
    elif roi_image.ndim == 3 and roi_image.shape[-1] == 3:
        ax.imshow(roi_image)
    else:
        ax.imshow(np.squeeze(roi_image), cmap="gray")

    h, w = int(roi_image.shape[0]), int(roi_image.shape[1])

    if show_bbox:
        if bbox_xywhn_pose is not None and bbox_xywhn_pose.shape[0] == 4 and np.isfinite(bbox_xywhn_pose).all():
            x, y, bw, bh = _xywhn_to_rect(bbox_xywhn_pose, width=w, height=h)
            rect = mpatches.Rectangle(
                (x, y), bw, bh, fill=False, edgecolor="lime", linewidth=1.8, linestyle="--"
            )
            ax.add_patch(rect)

    if show_keypoints and keypoints_xy.ndim == 2 and keypoints_xy.shape[1] == 2:
        for i in range(keypoints_xy.shape[0]):
            xy = keypoints_xy[i]
            if not np.isfinite(xy).all():
                continue
            ax.scatter([xy[0]], [xy[1]], s=24, c="lime", edgecolors="black", linewidths=0.6)
            if i < len(keypoint_labels):
                ax.text(
                    float(xy[0]) + 3.0,
                    float(xy[1]) + 3.0,
                    keypoint_labels[i],
                    color="yellow",
                    fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
                )

    ax.set_title(title)
    ax.set_axis_off()


def _decode_string_array(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="ignore"))
        else:
            out.append(str(value))
    return out


def _kp_bbox_xywhn_from_points(keypoints_xy: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    if keypoints_xy.ndim != 2 or keypoints_xy.shape[1] != 2 or keypoints_xy.shape[0] == 0:
        return None
    finite = np.isfinite(keypoints_xy).all(axis=1)
    if not np.any(finite):
        return None
    pts = keypoints_xy[finite]
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    span_x = max(0.0, x_max - x_min)
    span_y = max(0.0, y_max - y_min)
    # Match pose loader contract: bbox = span + 50% margin
    w_px = max(1e-6, span_x * 1.5)
    h_px = max(1e-6, span_y * 1.5)
    cx = np.clip((x_min + x_max) / 2.0, 0.0, float(width))
    cy = np.clip((y_min + y_max) / 2.0, 0.0, float(height))
    return np.array([cx / float(width), cy / float(height), w_px / float(width), h_px / float(height)], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Merged pose training .zarr path.")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Which split indices to browse.",
    )
    parser.add_argument("--crop-run", type=str, help="Crop run name (defaults to latest).")
    parser.add_argument("--keypoint-run", type=str, help="Keypoint run name (defaults to latest).")
    parser.add_argument("--start", type=int, default=0, help="Start position within selected split.")
    parser.add_argument("--limit", type=int, help="Limit number of selected samples.")
    parser.add_argument("--no-bbox", action="store_true", help="Disable bbox overlay.")
    parser.add_argument("--no-keypoints", action="store_true", help="Disable keypoint overlay.")
    parser.add_argument("--save-dir", type=Path, help="Directory to write rendered PNG files.")
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Render and save all selected samples to --save-dir.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not launch interactive window (useful with --save-all).",
    )
    args = parser.parse_args()

    root = zarr.open_group(str(args.zarr_path), mode="r")
    if "crop_runs" not in root or "keypoints_runs" not in root:
        raise SystemExit("Expected merged pose layout with crop_runs and keypoints_runs.")

    crop_run = _resolve_run(root["crop_runs"], args.crop_run)
    kp_run = _resolve_run(root["keypoints_runs"], args.keypoint_run)
    selected = _load_selected_indices(root, args.split)
    if args.limit is not None and args.limit >= 0:
        selected = selected[: int(args.limit)]
    if selected.size == 0:
        raise SystemExit("No samples selected.")

    crop_group = root[f"crop_runs/{crop_run}"]
    kp_group = root[f"keypoints_runs/{kp_run}"]

    roi_images = crop_group["roi_images"]
    bbox_pose_all = np.asarray(crop_group["bbox_norm_coords"][:], dtype=np.float32)
    if "crop_bbox_norm_coords" in crop_group:
        # Keep loading for schema enforcement/provenance sanity, but we do not render it.
        _ = np.asarray(crop_group["crop_bbox_norm_coords"][:], dtype=np.float32)
    else:
        print(
            "WARNING: merged crop run missing crop_bbox_norm_coords; "
            "continuing in legacy-compat mode."
        )
    keypoints_all = np.asarray(kp_group["keypoints_roi"][:], dtype=np.float32)
    detection_success = (
        np.asarray(kp_group["detection_success"][:], dtype=bool)
        if "detection_success" in kp_group
        else np.ones((keypoints_all.shape[0],), dtype=bool)
    )
    method_name = str(kp_group.attrs.get("method") or "").strip().lower()
    row_gate_applied = bool(kp_group.attrs.get("row_gate_applied", False))
    row_gate_policy = str(kp_group.attrs.get("row_gate_policy") or "").strip().lower()

    source_dataset_idx = (
        np.asarray(root["source_index/source_dataset_idx"][:], dtype=np.int64)
        if "source_index/source_dataset_idx" in root
        else None
    )
    source_frame_idx = (
        np.asarray(root["source_index/source_frame_idx"][:], dtype=np.int64)
        if "source_index/source_frame_idx" in root
        else None
    )
    source_dataset_id = (
        _decode_string_array(np.asarray(root["source_index/source_dataset_id"][:]))
        if "source_index/source_dataset_id" in root
        else []
    )
    detection_source = (
        np.asarray(crop_group["detection_source"][:], dtype=np.int64)
        if "detection_source" in crop_group
        else None
    )

    labels_raw = kp_group.attrs.get("keypoint_labels")
    if isinstance(labels_raw, (list, tuple)):
        keypoint_labels = [str(v) for v in labels_raw]
    else:
        keypoint_labels = [f"k{i}" for i in range(int(keypoints_all.shape[1]) if keypoints_all.ndim >= 2 else 0)]

    print(f"Merged Zarr: {args.zarr_path}")
    print(f"Crop run: {crop_run}")
    print(f"Keypoint run: {kp_run}")
    print(f"Split: {args.split}")
    print(f"Selected samples: {int(selected.shape[0])}")
    legacy_warning: Optional[str] = None
    if (
        method_name == "merged_export"
        and row_gate_applied
        and row_gate_policy == "refined_usable"
    ):
        selected_success = detection_success[selected] if selected.size > 0 else np.empty(0, dtype=bool)
        n_fail = int(np.sum(~selected_success))
        if n_fail > 0:
            legacy_warning = (
                "Legacy merged semantics detected: row_gate=refined_usable "
                f"but {n_fail} selected rows have detection_success=0. "
                "Re-export merged dataset with current exporter."
            )
            print(f"WARNING: {legacy_warning}")

    def build_title(split_pos: int, sample_idx: int) -> str:
        success = bool(detection_success[sample_idx]) if sample_idx < detection_success.shape[0] else False
        ds_text = "dataset=?"
        if source_dataset_idx is not None and sample_idx < source_dataset_idx.shape[0]:
            ds_idx = int(source_dataset_idx[sample_idx])
            ds_val = source_dataset_id[ds_idx] if 0 <= ds_idx < len(source_dataset_id) else f"idx:{ds_idx}"
            ds_text = f"dataset={ds_val}"
        frame_text = (
            f"source_frame={int(source_frame_idx[sample_idx])}"
            if source_frame_idx is not None and sample_idx < source_frame_idx.shape[0]
            else "source_frame=?"
        )
        src_text = (
            f"det_source={int(detection_source[sample_idx])}"
            if detection_source is not None and sample_idx < detection_source.shape[0]
            else "det_source=?"
        )
        return (
            f"split_pos={split_pos}/{int(selected.shape[0]) - 1} sample_idx={sample_idx} "
            f"success={int(success)}\n{ds_text} {frame_text} {src_text}"
        )

    def render_to_axis(ax, split_pos: int) -> None:
        split_pos = int(np.clip(split_pos, 0, selected.shape[0] - 1))
        sample_idx = int(selected[split_pos])
        roi = np.asarray(roi_images[sample_idx])
        bbox_pose = (
            bbox_pose_all[sample_idx]
            if sample_idx < bbox_pose_all.shape[0]
            else np.full((4,), np.nan, dtype=np.float32)
        )
        kp = (
            keypoints_all[sample_idx]
            if sample_idx < keypoints_all.shape[0]
            else np.full((0, 2), np.nan, dtype=np.float32)
        )
        roi_h, roi_w = int(roi.shape[0]), int(roi.shape[1])
        kp_bbox = _kp_bbox_xywhn_from_points(kp, width=roi_w, height=roi_h)
        _draw_sample(
            ax=ax,
            roi_image=roi,
            bbox_xywhn_pose=(bbox_pose if np.isfinite(bbox_pose).all() else kp_bbox),
            keypoints_xy=kp,
            show_bbox=not args.no_bbox,
            show_keypoints=not args.no_keypoints,
            keypoint_labels=keypoint_labels,
            title=build_title(split_pos, sample_idx),
        )

    if args.save_all:
        if args.save_dir is None:
            raise SystemExit("--save-all requires --save-dir.")
        args.save_dir.mkdir(parents=True, exist_ok=True)
        fig_save, ax_save = plt.subplots(1, 1, figsize=(6, 6))
        for split_pos in range(int(selected.shape[0])):
            render_to_axis(ax_save, split_pos)
            sample_idx = int(selected[split_pos])
            out_path = args.save_dir / f"{split_pos:06d}_sample_{sample_idx:06d}.png"
            fig_save.savefig(out_path, dpi=130, bbox_inches="tight")
            if (split_pos + 1) % 100 == 0:
                print(f"Saved {split_pos + 1}/{int(selected.shape[0])} frames...")
        plt.close(fig_save)
        print(f"Saved {int(selected.shape[0])} rendered frames to {args.save_dir}")

    if args.no_gui:
        return 0

    start_pos = int(np.clip(args.start, 0, selected.shape[0] - 1))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.22)
    render_to_axis(ax, start_pos)

    state = {"pos": start_pos}

    def on_slider(value) -> None:
        state["pos"] = int(value)
        render_to_axis(ax, state["pos"])
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.15, 0.12, 0.7, 0.03])
    slider = Slider(ax_slider, "Sample", 0, int(selected.shape[0]) - 1, valinit=start_pos, valstep=1)
    slider.on_changed(on_slider)

    ax_prev = plt.axes([0.15, 0.04, 0.1, 0.05])
    ax_next = plt.axes([0.27, 0.04, 0.1, 0.05])
    ax_prev10 = plt.axes([0.45, 0.04, 0.1, 0.05])
    ax_next10 = plt.axes([0.57, 0.04, 0.1, 0.05])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")
    btn_prev10 = Button(ax_prev10, "Prev 10")
    btn_next10 = Button(ax_next10, "Next 10")

    def set_pos(pos: int) -> None:
        slider.set_val(int(np.clip(pos, 0, selected.shape[0] - 1)))

    btn_prev.on_clicked(lambda _event: set_pos(state["pos"] - 1))
    btn_next.on_clicked(lambda _event: set_pos(state["pos"] + 1))
    btn_prev10.on_clicked(lambda _event: set_pos(state["pos"] - 10))
    btn_next10.on_clicked(lambda _event: set_pos(state["pos"] + 10))

    def on_key(event) -> None:
        if event.key == "left":
            set_pos(state["pos"] - 1)
        elif event.key == "right":
            set_pos(state["pos"] + 1)
        elif event.key == "pagedown":
            set_pos(state["pos"] - 10)
        elif event.key == "pageup":
            set_pos(state["pos"] + 10)

    fig.canvas.mpl_connect("key_press_event", on_key)
    title = "Merged Pose Zarr Viewer"
    if legacy_warning:
        title += " [LEGACY WARNING]"
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
