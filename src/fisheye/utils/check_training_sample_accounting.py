#!/usr/bin/env python3
"""Audit how raw rows become sampled train/val indices for training configs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import zarr

from fisheye.training.zarr_yolo_dataset_loader import GlobalIndexManager, ZarrDatasetConfig


def _count_by_path(indices: list[tuple[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for path, _idx in indices:
        counts[path] += 1
    return dict(counts)


def _safe_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _pose_breakdown(metadata) -> Dict[str, Any]:
    root = zarr.open(metadata.path, mode="r")
    kp_rows = None
    kp_success_true = None
    kp_run = metadata.keypoint_run
    if kp_run and "keypoints_runs" in root and kp_run in root["keypoints_runs"]:
        kp_group = root[f"keypoints_runs/{kp_run}"]
        row_gate_policy = str(kp_group.attrs.get("row_gate_policy") or "")
        row_gate_applied = bool(kp_group.attrs.get("row_gate_applied", False))
        if "keypoints_roi" in kp_group:
            kp_rows = int(kp_group["keypoints_roi"].shape[0])
        if "detection_success" in kp_group:
            success_arr = np.asarray(kp_group["detection_success"][:], dtype=bool)
            kp_success_true = int(np.sum(success_arr))
    else:
        row_gate_policy = ""
        row_gate_applied = False

    coords = np.asarray(root[metadata.bbox_array_path][:])
    bbox_valid_mask = ~np.isnan(coords[:, 0]) if coords.size else np.zeros(0, dtype=bool)
    bbox_valid_count = int(np.sum(bbox_valid_mask))

    source_real_mask = np.ones_like(bbox_valid_mask, dtype=bool)
    if (
        metadata.requested_source_type in ["filtered", "detect", "manual"]
        and metadata.has_interpolated
        and metadata.detection_source_path
    ):
        detection_source = np.asarray(root[metadata.detection_source_path][:], dtype=np.int64)
        source_real_mask = detection_source == 0
    source_real_count = int(np.sum(source_real_mask))

    pose_success_mask = np.ones_like(bbox_valid_mask, dtype=bool)
    if kp_run and "keypoints_runs" in root and kp_run in root["keypoints_runs"] and not row_gate_applied:
        kp_group = root[f"keypoints_runs/{kp_run}"]
        if "detection_success" in kp_group:
            pose_success_mask = np.asarray(kp_group["detection_success"][:], dtype=bool)
    pose_success_count = int(np.sum(pose_success_mask))

    final_mask = bbox_valid_mask & source_real_mask & pose_success_mask
    final_valid_count = int(np.sum(final_mask))

    return {
        "keypoint_rows": kp_rows,
        "keypoint_success_true": kp_success_true,
        "bbox_valid": bbox_valid_count,
        "source_real": source_real_count,
        "pose_success": pose_success_count,
        "final_valid": final_valid_count,
        "row_gate_policy": row_gate_policy or None,
        "row_gate_applied": bool(row_gate_applied),
    }


def _detect_breakdown(metadata) -> Dict[str, Any]:
    root = zarr.open(metadata.path, mode="r")
    coords = np.asarray(root[metadata.bbox_array_path][:])
    bbox_valid_mask = ~np.isnan(coords[:, 0]) if coords.size else np.zeros(0, dtype=bool)
    bbox_valid_count = int(np.sum(bbox_valid_mask))

    source_real_mask = np.ones_like(bbox_valid_mask, dtype=bool)
    if (
        metadata.requested_source_type in ["filtered", "detect", "manual"]
        and metadata.has_interpolated
        and metadata.detection_source_path
    ):
        detection_source = np.asarray(root[metadata.detection_source_path][:], dtype=np.int64)
        source_real_mask = detection_source == 0
    source_real_count = int(np.sum(source_real_mask))
    final_mask = bbox_valid_mask & source_real_mask
    final_valid_count = int(np.sum(final_mask))

    return {
        "keypoint_rows": None,
        "keypoint_success_true": None,
        "bbox_valid": bbox_valid_count,
        "source_real": source_real_count,
        "pose_success": None,
        "final_valid": final_valid_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit row accounting for training dataset configs.",
    )
    parser.add_argument("config", type=Path, help="Training config YAML used for train_detection/train_pose.")
    args = parser.parse_args()

    cfg = ZarrDatasetConfig.from_yaml(str(args.config))
    manager = GlobalIndexManager(cfg)
    train_indices, val_indices = manager.get_split_indices()

    train_counts = _count_by_path(train_indices)
    val_counts = _count_by_path(val_indices)

    task = str(cfg.task).lower()
    print(f"Task: {task}")
    print(f"Sampling strategy: {cfg.sampling_strategy.value}")
    print(f"Datasets: {len(manager.metadata_list)}")
    print(f"Global sampled indices: {len(manager.global_indices)}")
    print(f"Split counts: train={len(train_indices)} val={len(val_indices)}")
    print()

    header = [
        "dataset",
        "requested_source",
        "selected_keypoint_run",
        "row_gate_policy",
        "row_gate_applied",
        "total_frames",
        "keypoint_rows",
        "keypoint_success_true",
        "bbox_valid",
        "source_real",
        "pose_success",
        "final_valid",
        "sampled",
        "train",
        "val",
        "sampled/final_valid",
    ]
    print("\t".join(header))

    total_keypoint_rows = 0
    total_final_valid = 0
    total_sampled = 0
    total_train = 0
    total_val = 0

    for metadata in manager.metadata_list:
        if task == "pose":
            stats = _pose_breakdown(metadata)
        else:
            stats = _detect_breakdown(metadata)

        sampled = int(manager.selected_indices_by_path.get(metadata.path, np.empty(0, dtype=int)).size)
        train_n = int(train_counts.get(metadata.path, 0))
        val_n = int(val_counts.get(metadata.path, 0))
        final_valid = int(stats.get("final_valid") or 0)

        total_keypoint_rows += int(stats["keypoint_rows"] or 0)
        total_final_valid += final_valid
        total_sampled += sampled
        total_train += train_n
        total_val += val_n

        row = [
            metadata.name,
            str(metadata.requested_source_type),
            str(metadata.keypoint_run or "-"),
            str(stats.get("row_gate_policy") or "-"),
            str(int(bool(stats.get("row_gate_applied", False)))),
            str(metadata.total_frames),
            str(stats["keypoint_rows"] if stats["keypoint_rows"] is not None else "-"),
            str(stats["keypoint_success_true"] if stats["keypoint_success_true"] is not None else "-"),
            str(stats["bbox_valid"]),
            str(stats["source_real"]),
            str(stats["pose_success"] if stats["pose_success"] is not None else "-"),
            str(final_valid),
            str(sampled),
            str(train_n),
            str(val_n),
            f"{_safe_ratio(sampled, final_valid):.3f}" if final_valid > 0 else "-",
        ]
        print("\t".join(row))

    print()
    print("Totals")
    print(f"  keypoint_rows={total_keypoint_rows}")
    print(f"  final_valid={total_final_valid}")
    print(f"  sampled={total_sampled}")
    print(f"  train={total_train}")
    print(f"  val={total_val}")
    if total_final_valid > 0:
        print(f"  sampled/final_valid={_safe_ratio(total_sampled, total_final_valid):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
