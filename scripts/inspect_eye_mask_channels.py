#!/usr/bin/env python3
"""Visualize the channel breakdown for eye-mask training samples.

This script loads the eye-mask training configuration, builds the
train/val datasets, and renders the per-channel inputs (raw ROI,
optional binarized channel, optional Sobel/Laplacian/Canny edge map)
alongside the union mask used as the training target.

Usage examples:
    python scripts/inspect_eye_mask_channels.py configs/fisheye/eye_segmentation_config.yaml --split train --random 4
    python scripts/inspect_eye_mask_channels.py configs/fisheye/eye_segmentation_config.yaml --split val --indices 10 42 --save-dir runs/analysis/eye_mask_channels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from fisheye.training.config import EyeMaskTrainingConfig
from fisheye.training.zarr_eye_mask_dataset import build_eye_mask_datasets, EyeMaskYOLODataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect eye-mask training sample channels.")
    parser.add_argument("config", type=Path, help="Path to eye segmentation training config YAML.")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split to sample from (default: train).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Specific sample indices to visualize (overrides --random).",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=3,
        help="Number of random samples to visualize when --indices is not provided (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling indices (default: 0).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Directory to save figures. If omitted, figures are shown interactively.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="DPI for saved figures (default: 120).",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable mask visualizations (mask and overlay panels).",
    )
    return parser.parse_args()


def _build_channel_labels(cfg: EyeMaskTrainingConfig, channel_count: int) -> List[str]:
    labels: List[str] = ["raw"]
    if cfg.binarization_threshold is not None:
        labels.append(f"binary>={cfg.binarization_threshold}")
    if cfg.edge_enhancement:
        labels.append(f"{cfg.edge_enhancement}")
    while len(labels) < channel_count:
        labels.append(f"ch{len(labels)}")
    return labels[:channel_count]


def _select_indices(dataset: EyeMaskYOLODataset, indices: Iterable[int] | None, random_count: int, seed: int) -> List[int]:
    if indices is not None:
        result = [idx for idx in indices if 0 <= idx < len(dataset)]
        if not result:
            raise ValueError("No valid indices were provided.")
        return sorted(set(result))

    rng = np.random.default_rng(seed)
    random_count = max(1, min(random_count, len(dataset)))
    selection = rng.choice(len(dataset), size=random_count, replace=False)
    return sorted(int(idx) for idx in selection)


def _ensure_save_dir(save_dir: Path | None) -> Path | None:
    if save_dir is None:
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _plot_sample(
    dataset: EyeMaskYOLODataset,
    cfg: EyeMaskTrainingConfig,
    index: int,
    include_mask: bool,
    save_dir: Path | None,
    dpi: int,
) -> None:
    sample = dataset[index]
    image = sample["img"]  # shape: (C, H, W)
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    channels, height, width = image.shape
    channel_labels = _build_channel_labels(cfg, channels)

    mask_tensor = sample.get("masks")
    mask_array = None
    mask_labels = sample.get("mask_labels")
    if include_mask and mask_tensor is not None:
        mask_array = np.asarray(mask_tensor)
        if mask_array.ndim == 2:
            mask_array = mask_array[None, ...]
    mask_count = mask_array.shape[0] if mask_array is not None else 0

    extra_cols = 0
    if include_mask:
        extra_cols = mask_count + 1 if mask_count > 0 else 2
    n_cols = channels + extra_cols
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]  # type: ignore[assignment]
    else:
        axes = list(np.ravel(axes))

    for ch in range(channels):
        ax = axes[ch]
        channel_img = image[ch]
        ax.imshow(channel_img, cmap="gray")
        ax.set_title(channel_labels[ch])
        ax.axis("off")

    entry = dataset.entries[index]
    suptitle = f"{entry.dataset_name} | split={dataset.split_name} | idx={index} | type={entry.entry_type}"
    if entry.entry_type == "roi":
        suptitle += f" | has_positive={entry.has_positive}"
    fig.suptitle(suptitle, fontsize=12)

    if include_mask:
        overlay_colors = [
            (1.0, 0.2, 0.2),  # red-ish
            (0.2, 0.6, 1.0),  # blue-ish
            (0.4, 1.0, 0.4),  # green-ish
            (1.0, 0.6, 0.2),  # orange-ish
        ]
        overlay_base = np.stack([image[0], image[0], image[0]], axis=-1).astype(np.float32)
        denom = float(overlay_base.max() - overlay_base.min()) or 1.0
        overlay_rgb = (overlay_base - overlay_base.min()) / denom

        if mask_array is not None and mask_count > 0:
            offset = channels
            for mask_idx in range(mask_count):
                mask_slice = mask_array[mask_idx]
                ax_mask = axes[offset]
                label_val = None
                if mask_labels is not None and mask_idx < len(mask_labels):
                    label_val = int(mask_labels[mask_idx])
                title_suffix = f"mask_{label_val}" if label_val is not None else f"mask_{mask_idx}"
                ax_mask.imshow(mask_slice, cmap="gray")
                ax_mask.set_title(title_suffix)
                ax_mask.axis("off")
                color = overlay_colors[mask_idx % len(overlay_colors)]
                mask_bool = mask_slice > 0.5
                overlay_rgb[mask_bool, 0] = color[0]
                overlay_rgb[mask_bool, 1] = color[1]
                overlay_rgb[mask_bool, 2] = color[2]
                offset += 1
        else:
            offset = channels
            ax_empty = axes[offset]
            ax_empty.imshow(np.zeros((height, width), dtype=np.float32), cmap="gray")
            ax_empty.set_title("mask_none")
            ax_empty.axis("off")
            offset += 1

        ax_overlay = axes[-1]
        ax_overlay.imshow(overlay_rgb)
        ax_overlay.set_title("raw + masks")
        ax_overlay.axis("off")

    plt.tight_layout()

    if save_dir is not None:
        filename = f"{entry.dataset_name}_{dataset.split_name}_{index}.png"
        fig.savefig(save_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = _parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = EyeMaskTrainingConfig.from_yaml(args.config)
    bundle = build_eye_mask_datasets(cfg)
    dataset = bundle.train_dataset if args.split == "train" else bundle.val_dataset

    indices = _select_indices(dataset, args.indices, args.random, args.seed)
    save_dir = _ensure_save_dir(args.save_dir)
    include_mask = not args.no_mask

    for idx in indices:
        _plot_sample(
            dataset=dataset,
            cfg=cfg,
            index=idx,
            include_mask=include_mask,
            save_dir=save_dir,
            dpi=args.dpi,
        )

        if save_dir is None:
            response = input("Press Enter for next sample, or 'q' to quit: ").strip().lower()
            if response == "q":
                break

    if save_dir is not None:
        print(f"Saved {len(indices)} figure(s) to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
