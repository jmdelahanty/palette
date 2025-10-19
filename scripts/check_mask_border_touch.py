#!/usr/bin/env python3
"""Report how often ground-truth eye masks touch ROI borders.

This helps decide whether ROI padding should be increased before resizing.
For each sample in the selected split, the script checks every mask channel
and tracks instances where any positive pixel lies on the outer border.
It reports per-channel counts (based on ``mask_labels`` when available)
and the overall fraction of masks touching the border.

Example usage:

    python scripts/check_mask_border_touch.py configs/fisheye/eye_segmentation_config.yaml \
        --split train --max-samples 10000 --threshold 0.2
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from fisheye.training.config import EyeMaskTrainingConfig
from fisheye.training.zarr_eye_mask_dataset import build_eye_mask_datasets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check border-touch rate for eye mask channels.")
    parser.add_argument("config", type=Path, help="Path to eye segmentation training config YAML.")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split to evaluate (default: train).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to scan (default: entire split).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Warn if border-touch rate exceeds this fraction (default: 0.2 == 20%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling when --max-samples is set.",
    )
    return parser.parse_args()


def _mask_touches_border(mask: np.ndarray) -> bool:
    if mask.sum() == 0:
        return False
    if (mask[0, :] > 0).any() or (mask[-1, :] > 0).any():
        return True
    if (mask[:, 0] > 0).any() or (mask[:, -1] > 0).any():
        return True
    return False


def main() -> None:
    args = _parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = EyeMaskTrainingConfig.from_yaml(args.config)
    bundle = build_eye_mask_datasets(cfg)
    dataset = bundle.train_dataset if args.split == "train" else bundle.val_dataset

    total_masks = 0
    touching_masks = 0
    per_label_counts: Counter[int] = Counter()
    per_label_touch: Counter[int] = Counter()

    indices = np.arange(len(dataset))
    if args.max_samples is not None and args.max_samples < len(indices):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(indices, size=args.max_samples, replace=False)

    for idx in indices:
        sample = dataset[idx]
        mask_tensor = sample.get("masks")
        if mask_tensor is None:
            continue
        masks = np.asarray(mask_tensor)
        if masks.ndim == 2:
            masks = masks[None, ...]
        labels = sample.get("mask_labels")
        for mask_idx, mask_ch in enumerate(masks):
            total_masks += 1
            label_val = None
            if labels is not None and mask_idx < len(labels):
                try:
                    label_val = int(labels[mask_idx])
                except (TypeError, ValueError):
                    label_val = None
            if label_val is not None:
                per_label_counts[label_val] += 1
            if _mask_touches_border(mask_ch > 0.5):
                touching_masks += 1
                if label_val is not None:
                    per_label_touch[label_val] += 1

    if total_masks == 0:
        print("No mask channels found in the selected split.")
        return

    overall_rate = touching_masks / total_masks
    print(f"Scanned {len(indices)} samples, {total_masks} mask channels.")
    print(f"Overall border-touch count: {touching_masks}")
    print(f"Overall rate: {overall_rate:.3%}")

    if per_label_counts:
        print("\nPer-label statistics:")
        for label, count in sorted(per_label_counts.items()):
            touch = per_label_touch.get(label, 0)
            rate = touch / count if count else 0.0
            print(f"  label {label}: {touch}/{count} ({rate:.3%})")

    if overall_rate >= args.threshold:
        print(
            f"\n[!] Border-touch rate {overall_rate:.2%} exceeds threshold {args.threshold:.2%}. "
            "Consider increasing ROI padding by ~4–8 pixels before resizing."
        )
    else:
        print(f"\nBorder-touch rate is below threshold {args.threshold:.2%}.")


if __name__ == "__main__":
    main()

