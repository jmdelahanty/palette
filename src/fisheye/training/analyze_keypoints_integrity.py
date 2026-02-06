"""
Analyze keypoint integrity for pose training datasets.

This script inspects the Zarr keypoints arrays for each dataset declared in a pose_config
and reports:
  - Total ROIs, detection_success counts and rate
  - Rows with all-NaN keypoints vs. any-NaN keypoints
  - Per-keypoint missing rates (among successful rows)
  - Out-of-bounds rates (keypoints outside ROI size, among successful rows)
  - Count of indices that would be used for training (matches our loader logic)

Usage:
  python -m fisheye.training.analyze_keypoints_integrity configs/fisheye/pose_config.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import zarr

from .config import PoseConfig
from .zarr_yolo_dataset_loader import ZarrDatasetConfig, GlobalIndexManager


def build_zarr_config(full_config: PoseConfig) -> ZarrDatasetConfig:
    datasets_dict: Dict[str, Dict[str, Any]] = {}
    for name, ds_cfg in full_config.datasets.items():
        split_dict = None
        if ds_cfg.split is not None:
            split_dict = {"train": ds_cfg.split.train, "val": ds_cfg.split.val}
        datasets_dict[name] = {
            "zarr_path": str(ds_cfg.zarr_path),
            "source_type": ds_cfg.source_type.value if hasattr(ds_cfg.source_type, "value") else ds_cfg.source_type,
            "input_format": ds_cfg.input_format,
            "keypoint_run": ds_cfg.keypoint_run,
            "split": split_dict,
        }

    default_split = 0.8
    if full_config.datasets:
        first_ds = next(iter(full_config.datasets.values()))
        if first_ds.split is not None:
            default_split = first_ds.split.train

    return ZarrDatasetConfig(
        datasets=datasets_dict,
        task=full_config.task,
        sampling_strategy=full_config.sampling_strategy.value if hasattr(full_config.sampling_strategy, "value") else full_config.sampling_strategy,
        random_seed=full_config.random_seed,
        dataset_weights=full_config.dataset_weights,
        split_ratio=default_split,
    )


@dataclass
class KpIntegrityReport:
    name: str
    total_rois: int
    success_true: int
    success_false: int
    all_nan_rows: int
    any_nan_rows: int
    per_kpt_missing_rate: List[float]
    out_of_bounds_rate: float
    used_for_training: int


def analyze_one_zarr(path: str) -> KpIntegrityReport:
    root = zarr.open(path, mode="r")

    # ROI size from crop run
    latest_crop = root["crop_runs"].attrs["latest"]
    crop_group = root[f"crop_runs/{latest_crop}"]
    roi_h, roi_w = crop_group["roi_images"].shape[1:3]

    # Keypoints
    latest_kp = root["keypoints_runs"].attrs["latest"]
    kp_group = root[f"keypoints_runs/{latest_kp}"]
    kp = kp_group["keypoints_roi"][:]  # (N,K,2) in ROI pixels
    success = kp_group["detection_success"][:]  # (N,)
    labels = list(kp_group.attrs.get("keypoint_labels", []))
    K = kp.shape[1] if kp.ndim == 3 else 0

    N = kp.shape[0]
    is_nan = np.isnan(kp)
    # Rows with every value NaN
    all_nan_rows = (is_nan.reshape(N, -1).sum(axis=1) == (K * 2)).sum()
    # Rows with any NaN
    any_nan_rows = (is_nan.reshape(N, -1).sum(axis=1) > 0).sum()

    # Per-keypoint missing rate among successful rows
    per_kpt_missing = []
    suc_idx = np.where(success)[0]
    if suc_idx.size > 0 and K > 0:
        for k in range(K):
            miss_k = np.isnan(kp[suc_idx, k, :]).any(axis=1)
            per_kpt_missing.append(float(miss_k.mean()) * 100.0)
    else:
        per_kpt_missing = [0.0] * K

    # Out-of-bounds among successful rows
    if suc_idx.size > 0 and roi_w > 0 and roi_h > 0:
        xs = kp[suc_idx, :, 0]
        ys = kp[suc_idx, :, 1]
        oob = (xs < 0) | (ys < 0) | (xs > (roi_w - 1)) | (ys > (roi_h - 1))
        oob_rows = oob.any(axis=1)
        out_of_bounds_rate = float(oob_rows.mean()) * 100.0
    else:
        out_of_bounds_rate = 0.0

    # How many samples would be used by our training split? Reuse index manager logic
    # Note: this uses dataset-wide settings (filtered/detect) and success mask.
    # We build a temporary config with only this Zarr path.
    zcfg = ZarrDatasetConfig(
        datasets={
            str(path): {
                "zarr_path": path,
                "source_type": "filtered",
                "split": None,
            }
        },
        task="pose",
        split_ratio=0.8,
    )
    gidx = GlobalIndexManager(zcfg)
    # Count valid indices for this path
    used_for_training = sum(1 for p, _ in gidx.global_indices if p == path)

    return KpIntegrityReport(
        name=str(path),
        total_rois=N,
        success_true=int(success.sum()),
        success_false=int((~success).sum()),
        all_nan_rows=int(all_nan_rows),
        any_nan_rows=int(any_nan_rows),
        per_kpt_missing_rate=per_kpt_missing,
        out_of_bounds_rate=out_of_bounds_rate,
        used_for_training=used_for_training,
    )


def main():
    ap = argparse.ArgumentParser(description="Analyze keypoints integrity for pose training")
    ap.add_argument("config", type=str, help="Path to pose_config.yaml")
    args = ap.parse_args()

    full = PoseConfig.from_yaml(args.config)
    zcfg = build_zarr_config(full)
    paths = zcfg.get_zarr_paths()

    print("=== Keypoints Integrity Report ===")
    for p in paths:
        rep = analyze_one_zarr(p)
        print(f"\nZarr: {rep.name}")
        print(f"  Total ROIs: {rep.total_rois}")
        print(f"  detection_success: {rep.success_true} true | {rep.success_false} false"
              f" ({rep.success_true / max(rep.total_rois,1) * 100.0:.2f}% success)")
        print(f"  Rows with all-NaN keypoints: {rep.all_nan_rows}")
        print(f"  Rows with any-NaN keypoint: {rep.any_nan_rows}")
        if rep.per_kpt_missing_rate:
            labels = " / ".join([f"k{k}" for k in range(len(rep.per_kpt_missing_rate))])
            print(f"  Per-kpt missing (success rows): {labels}")
            print("   ", ", ".join([f"{v:.2f}%" for v in rep.per_kpt_missing_rate]))
        print(f"  Out-of-bounds rate (success rows): {rep.out_of_bounds_rate:.2f}%")
        print(f"  Valid indices used for training/val: {rep.used_for_training}")


if __name__ == "__main__":
    main()
