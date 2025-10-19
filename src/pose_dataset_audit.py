# src/pose_dataset_audit.py
"""Audit the pose training dataset and flag frames that will yield empty labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr

from fisheye.training.zarr_yolo_dataset_loader import (
    ZarrDatasetConfig,
    create_zarr_dataset,
)


def audit_pose_dataset(
    zarr_path: Path,
    dataset_alias: str | None = None,
    split_ratio: float = 0.8,
    random_seed: int = 42,
) -> None:
    """Audit pose dataset selection and highlight samples that produce empty labels."""

    print("🔍 Pose Dataset Audit")
    print(f"📁 Zarr: {zarr_path}")

    if not zarr_path.exists():
        print("❌ Zarr path does not exist.")
        return

    # Build a minimal config: single dataset with chosen keypoint run logic
    dataset_name = dataset_alias or zarr_path.stem
    config = ZarrDatasetConfig(
        datasets={
            dataset_name: {
                "zarr_path": str(zarr_path),
                "source_type": "filtered",
                "keypoint_run": "latest_traditional",
            }
        },
        task="pose",
        split_ratio=split_ratio,
        random_seed=random_seed,
    )

    # Instantiate dataset once for metadata
    dataset = create_zarr_dataset(config=config, mode="train")

    total_samples = len(dataset)
    print(f"📊 Total samples after global filtering: {total_samples}")

    empty_indices: List[int] = []

    for idx in range(total_samples):
        sample = dataset[idx]
        if sample["cls"].size == 0 or sample["keypoints"].size == 0:
            zarr_src, roi_idx = dataset.indices[idx]
            empty_indices.append((idx, Path(zarr_src).stem, roi_idx))

    if empty_indices:
        print(f"\n⚠️ Found {len(empty_indices)} samples that yield empty labels.")
        for i, (ds_idx, src_name, roi_idx) in enumerate(empty_indices[:20], start=1):
            print(f"  {i:02d}. dataset_idx={ds_idx}, zarr={src_name}, roi_idx={roi_idx}")
        if len(empty_indices) > 20:
            print("  …")
    else:
        print("\n✅ No empty-label samples detected.")

    print("\n🎯 Summary")
    print("---------")
    print(f"Total samples inspected : {total_samples}")
    print(f"Empty-label samples     : {len(empty_indices)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pose dataset selection.")
    parser.add_argument("zarr_path", type=Path, help="Path to a Palette Zarr store.")
    parser.add_argument("--dataset-name", help="Optional alias for the dataset entry.")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used for splitting.")
    args = parser.parse_args()

    audit_pose_dataset(
        zarr_path=args.zarr_path,
        dataset_alias=args.dataset_name,
        split_ratio=args.split_ratio,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
