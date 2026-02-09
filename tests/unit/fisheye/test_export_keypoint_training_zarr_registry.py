"""Tests for keypoint merged-export registry linkage updates."""

import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils.export_keypoint_training_zarr import _register_merged_dataset_in_registry


def test_register_merged_keypoint_dataset_updates_training_set_linkage(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    merged_zarr = tmp_path / "merged_pose_training.zarr"

    root = zarr.open_group(str(merged_zarr), mode="w")
    root.attrs["session_uuid"] = "pose_my_set_v001_merged"
    root.attrs["zarr_purpose"] = "training"
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "merged_export_smoke"
    crop = crop_parent.create_group("merged_export_smoke")
    crop.create_array(
        "roi_images",
        data=np.zeros((2, 8, 8), dtype=np.uint8),
        chunks=(1, 8, 8),
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.zeros((2, 4), dtype=np.float32),
        chunks=(2, 4),
    )
    kp_parent = root.create_group("keypoints_runs")
    kp_parent.attrs["latest"] = "merged_export_smoke"
    kp = kp_parent.create_group("merged_export_smoke")
    kp.create_array(
        "keypoints_roi",
        data=np.zeros((2, 3, 2), dtype=np.float32),
        chunks=(2, 3, 2),
    )
    kp.create_array(
        "detection_success",
        data=np.array([True, True], dtype=np.bool_),
        chunks=(2,),
    )

    registry = Registry(registry_path)
    registry.upsert_dataset(
        "source_a",
        session_uuid="source_a",
        zarr_path=tmp_path / "source_a.zarr",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "source_b",
        session_uuid="source_b",
        zarr_path=tmp_path / "source_b.zarr",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_training_set(
        set_id="pose_my_set_v001",
        name="my_set",
        query_filter={"tool": "fisheye.utils.prepare_keypoint_training_from_registry"},
        dataset_ids=["source_a", "source_b"],
        invocation={"tool": "fisheye.utils.prepare_keypoint_training_from_registry"},
    )
    registry.close()

    merged_dataset_id, linked = _register_merged_dataset_in_registry(
        registry_path=registry_path,
        merged_zarr=merged_zarr,
        set_id="pose_my_set_v001",
        set_name="my_set",
        source_dataset_ids=["source_a", "source_b"],
        query_filter={"tool": "fisheye.utils.prepare_keypoint_training_from_registry"},
        invocation={"tool": "fisheye.utils.prepare_keypoint_training_from_registry"},
    )

    assert merged_dataset_id == "pose_my_set_v001_merged"
    assert linked is True

    db = Registry(registry_path)
    dataset_row = db.conn.execute(
        "SELECT dataset_id, zarr_path FROM datasets WHERE dataset_id = ?",
        ("pose_my_set_v001_merged",),
    ).fetchone()
    assert dataset_row is not None
    assert str(merged_zarr) == dataset_row["zarr_path"]

    set_row = db.conn.execute(
        "SELECT dataset_ids_json FROM training_sets WHERE set_id = ?",
        ("pose_my_set_v001",),
    ).fetchone()
    assert set_row is not None
    dataset_ids = json.loads(set_row["dataset_ids_json"])
    assert dataset_ids == sorted(["source_a", "source_b", "pose_my_set_v001_merged"])
    db.close()
