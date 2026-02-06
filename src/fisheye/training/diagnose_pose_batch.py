"""
Diagnostic script for inspecting pose/keypoint batches.

Usage:
  python -m fisheye.training.diagnose_pose_batch configs/fisheye/pose_config.yaml [--batch-size 8]

This loads the validation split using the same dataset + collate_fn
as training, fetches one batch, and prints shapes and basic stats.
"""

from __future__ import annotations

import argparse
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PoseConfig
from .zarr_yolo_dataset_loader import ZarrDatasetConfig, create_zarr_dataset
from .train_keypoints import pose_collate_fn


def build_zarr_config(full_config: PoseConfig) -> ZarrDatasetConfig:
    """Build ZarrDatasetConfig from PoseConfig, mirroring train_keypoints.py."""
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

    # Set target_size to training imgsz to avoid surprises
    target_size = getattr(full_config.training_params, "imgsz", None)

    return ZarrDatasetConfig(
        datasets=datasets_dict,
        task=full_config.task,
        sampling_strategy=full_config.sampling_strategy.value if hasattr(full_config.sampling_strategy, "value") else full_config.sampling_strategy,
        random_seed=full_config.random_seed,
        dataset_weights=full_config.dataset_weights,
        split_ratio=default_split,
        target_size=target_size,
    )


def summarize_dataset(ds) -> Dict[str, Any]:
    """Return quick stats based on pre-cached labels."""
    n_samples = len(ds)
    n_instances_total = 0
    n_empty = 0
    for lbl in ds.labels:
        cls = lbl.get("cls", np.zeros((0,), dtype=np.float32))
        n_instances_total += int(len(cls))
        if len(cls) == 0:
            n_empty += 1
    return {
        "samples": n_samples,
        "instances_total": n_instances_total,
        "empty_samples": n_empty,
        "keypoint_labels": getattr(ds, "keypoint_labels", []),
    }


def summarize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize tensor shapes and a few values from a collated batch."""
    out: Dict[str, Any] = {}
    img = batch.get("img")
    out["img_shape"] = tuple(img.shape) if isinstance(img, torch.Tensor) else None

    batch_idx = batch.get("batch_idx")
    if isinstance(batch_idx, torch.Tensor):
        out["batch_idx_shape"] = tuple(batch_idx.shape)
        out["batch_idx_unique"] = batch_idx.unique().cpu().tolist()
    else:
        out["batch_idx_shape"] = None
        out["batch_idx_unique"] = None

    cls = batch.get("cls")
    if isinstance(cls, torch.Tensor):
        out["cls_shape"] = tuple(cls.shape)
        out["cls_dtype"] = str(cls.dtype)
        # sample first up to 10 values
        out["cls_sample"] = cls[:10].cpu().tolist()
    else:
        out["cls_shape"] = None
        out["cls_dtype"] = None
        out["cls_sample"] = None

    bboxes = batch.get("bboxes")
    if isinstance(bboxes, torch.Tensor):
        out["bboxes_shape"] = tuple(bboxes.shape)
        out["bboxes_sample"] = bboxes[:3].cpu().tolist()
    else:
        out["bboxes_shape"] = None
        out["bboxes_sample"] = None

    kpts = batch.get("keypoints")
    if isinstance(kpts, torch.Tensor):
        out["keypoints_shape"] = tuple(kpts.shape)
        # sample first instance if present
        if kpts.numel() > 0 and kpts.ndim >= 3:
            out["keypoints_first_instance"] = kpts[0].cpu().tolist()
        else:
            out["keypoints_first_instance"] = None
    else:
        out["keypoints_shape"] = None
        out["keypoints_first_instance"] = None

    out["im_file_len"] = len(batch.get("im_file", []))
    out["ori_shape_len"] = len(batch.get("ori_shape", []))
    return out


def main():
    ap = argparse.ArgumentParser(description="Diagnose pose/keypoint validation batch")
    ap.add_argument("config", type=str, help="Path to pose_config.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    # Load config
    full_config = PoseConfig.from_yaml(args.config)
    zcfg = build_zarr_config(full_config)

    # Build validation dataset and loader
    val_ds = create_zarr_dataset(config=zcfg, mode="val")
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pose_collate_fn)

    ds_stats = summarize_dataset(val_ds)
    print("=== Dataset Summary ===")
    print(f"Samples (val): {ds_stats['samples']}")
    print(f"Instances total (sum of cls): {ds_stats['instances_total']}")
    print(f"Empty-label samples: {ds_stats['empty_samples']}")
    print(f"Keypoint labels: {ds_stats['keypoint_labels']}")
    print(f"Config kpt_shape: {full_config.kpt_shape}")

    # Get one batch
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("No samples in validation dataset.")
        return

    batch_stats = summarize_batch(batch)
    print("\n=== Batch Summary ===")
    for k, v in batch_stats.items():
        print(f"{k}: {v}")

    # Heuristic checks
    print("\n=== Heuristic Checks ===")
    kpt_shape = batch_stats.get("keypoints_shape")
    if kpt_shape and len(kpt_shape) == 3:
        num_kpts = kpt_shape[1]
        dims_per_kpt = kpt_shape[2]
        print(f"Keypoints per instance: {num_kpts}, dims per keypoint: {dims_per_kpt}")
        # Compare to YAML
        try:
            yaml_k, yaml_d = full_config.kpt_shape
            if yaml_k != num_kpts or yaml_d != dims_per_kpt:
                print(
                    f"WARNING: kpt_shape mismatch. YAML={full_config.kpt_shape} vs batch={(num_kpts, dims_per_kpt)}"
                )
        except Exception:
            pass
    else:
        print("No keypoints present in batch or unexpected keypoint shape.")

    # Sanity: instances per batch should match len(cls)
    cls_shape = batch_stats.get("cls_shape")
    if cls_shape is not None and len(cls_shape) == 1:
        num_instances = cls_shape[0]
        if batch_stats.get("bboxes_shape") and len(batch_stats["bboxes_shape"]) == 2:
            if batch_stats["bboxes_shape"][0] != num_instances:
                print(
                    f"WARNING: bboxes count {batch_stats['bboxes_shape'][0]} != cls count {num_instances}"
                )
        if kpt_shape and len(kpt_shape) == 3 and kpt_shape[0] != num_instances:
            print(
                f"WARNING: keypoints instance count {kpt_shape[0]} != cls count {num_instances}"
            )


if __name__ == "__main__":
    main()
