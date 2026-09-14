"""End-to-end materialized pose source export; run outside the sandbox."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.materialized_pose_training_source import (
    inspect_materialized_pose_crop_source,
)
from fisheye.utils.export_keypoint_training_zarr import (
    _export_merged,
    _write_merged_config,
    validate_merged_keypoint_training_zarr,
)
from fisheye.training.config import PoseConfig
from fisheye.training.zarr_yolo_dataset_loader import (
    ZarrDatasetConfig,
    ZarrYOLODataset,
)


RUN_ID = "head_crop_v001"
LABELS = ["swim_bladder", "eye_left", "eye_right"]
EDGES = [[0, 1], [0, 2], [1, 2]]


def _digest(values: np.ndarray) -> str:
    packed = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(packed.dtype).encode("ascii"))
    digest.update(json.dumps(packed.shape).encode("ascii"))
    digest.update(packed.tobytes())
    return digest.hexdigest()


def _source(path: Path, *, pixel_value: int) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["zarr_purpose"] = "training"
    raw = root.create_group("raw_video")
    raw.create_array("original_frame_indices", data=np.array([10, 20], dtype=np.int64))
    raw.create_array("images_full", data=np.zeros((2, 128, 128), dtype=np.uint8))
    crop_parent = root.create_group("crop_runs")
    source_crop = crop_parent.create_group("original_crop")
    source_crop.attrs["detection_source_type"] = "refined"
    source_crop.attrs["palette_run_completion_status"] = "complete"
    source_crop.create_array("roi_images", data=np.zeros((5, 64, 64), dtype=np.uint8))
    keypoint_parent = root.create_group("keypoints_runs")
    reviewed = keypoint_parent.create_group("reviewed_pose")
    reviewed.attrs["palette_run_completion_status"] = "complete"
    reviewed.create_array("keypoints_roi", data=np.zeros((3, 3, 2), dtype=np.float32))
    run = crop_parent.create_group(RUN_ID)
    pixels = np.full((2, 64, 64), pixel_value, dtype=np.uint8)
    points = np.array(
        [
            [[24, 32], [16, 16], [32, 16]],
            [[25, 33], [17, 17], [33, 17]],
        ],
        dtype=np.float32,
    )
    visibility = np.full((2, 3), 2, dtype=np.uint8)
    for name, values in {
        "roi_images": pixels,
        "keypoints_roi": points,
        "keypoint_visibility": visibility,
        "roi_coordinates_full": np.array([[32, 32], [32, 32]], dtype=np.int32),
        "source_keypoint_row_ids": np.array([1, 2], dtype=np.int64),
        "source_crop_row_ids": np.array([3, 4], dtype=np.int64),
        "source_training_row_indices": np.array([0, 1], dtype=np.int64),
        "source_frame_indices": np.array([10, 20], dtype=np.int64),
        "bbox_img_xyxy": np.array(
            [[40, 40, 80, 80], [41, 41, 81, 81]], dtype=np.float32
        ),
    }.items():
        run.create_array(name, data=values)
    run.attrs.update(
        {
            "schema_id": "palette.training.pose_head_materialized_crop",
            "schema_version": 2,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "crop_storage_mode": "materialized",
            "roi_size": [64, 64],
            "pose_schema": {
                "name": "traditional_v1",
                "skeleton_id": "pose_schema:traditional_v1",
                "keypoint_labels": LABELS,
            },
            "source_bindings": {
                "source_crop_run": "original_crop",
                "source_keypoint_group": "keypoints_runs",
                "source_keypoint_run": "reviewed_pose",
                "source_images_path": "raw_video/images_full",
                "source_images_shape": [2, 128, 128],
            },
            "pixel_sha256": _digest(pixels),
            "keypoints_roi_sha256": _digest(points),
            "keypoint_visibility_sha256": _digest(visibility),
        }
    )


def _manifest(paths: list[Path]) -> dict:
    contents = []
    for path in paths:
        root = zarr.open_group(str(path), mode="r", use_consolidated=False)
        source = inspect_materialized_pose_crop_source(
            root[f"crop_runs/{RUN_ID}"], run_id=RUN_ID
        )
        contents.append(
            {
                "schema_id": source.schema_id,
                "schema_version": source.schema_version,
                "pixel_sha256": source.pixel_sha256,
                "keypoints_roi_sha256": source.keypoints_roi_sha256,
                "keypoint_visibility_sha256": source.keypoint_visibility_sha256,
                "source_row_lineage_sha256": source.source_row_lineage_sha256,
            }
        )
    return {
        "set_id": "pose_materialized_reuse_v001",
        "set_name": "materialized_reuse",
        "input_format": "gray",
        "source_type": "materialized_pose_crop",
        "pose_schema": {
            "skeleton_id": "pose_schema:traditional_v1",
            "kpt_shape": [3, 3],
            "keypoint_labels": LABELS,
            "skeleton": EDGES,
        },
        "datasets": [
            {
                "name": f"source_{index}",
                "dataset_id": f"source_{index}",
                "recording_id": f"recording_{index}",
                "zarr_path": str(path),
                "source_kind": "materialized_pose_crop",
                "materialized_crop_run": RUN_ID,
                "materialized_content": contents[index],
                "source_type_resolved": "materialized_pose_crop",
                "leakage_group": {
                    "id": f"subject:fish_{index}",
                    "source": "registered_subject",
                },
            }
            for index, path in enumerate(paths)
        ],
    }


def test_materialized_sources_use_existing_immutable_merge_contract(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "a.zarr", tmp_path / "b.zarr"]
    for index, path in enumerate(paths):
        _source(path, pixel_value=index + 1)
    manifest = _manifest(paths)
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "merged.zarr"
    result = _export_merged(
        manifest_payload=manifest,
        manifest_path=manifest_path,
        out_zarr=output,
        merged_dataset_id="pose_materialized_reuse_v001_merged",
        overwrite=False,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=7,
        copy_batch_size=2,
        row_gate_policy="auto",
        invocation={},
        roi_transform_mode="strict",
        target_roi_hw=(64, 64),
        split_unit="leakage_group",
        use_storage_contract_v2=True,
    )
    assert result.total_samples == 4
    validate_merged_keypoint_training_zarr(output, expected_total_samples=4)
    root = zarr.open_group(str(output), mode="a", use_consolidated=False)
    index = root["source_index"]
    np.testing.assert_array_equal(index["source_keypoint_row_id"][:], [1, 2, 1, 2])
    np.testing.assert_array_equal(index["source_crop_row_id"][:], [3, 4, 3, 4])
    np.testing.assert_array_equal(index["source_sample_row_index"][:], [0, 1, 0, 1])
    np.testing.assert_array_equal(
        index["source_acquisition_frame_index"][:], [10, 20, 10, 20]
    )
    np.testing.assert_array_equal(index["source_refined_row_ids"][:], [-1] * 4)
    assert root.attrs["training_export"]["materialized_pose_source_contract"][
        "all_keypoints_visible_required"
    ]
    assert len(root.attrs["training_export"]["logical_dataset_hash"]["digest"]) == 64
    loader = ZarrYOLODataset(
        ZarrDatasetConfig(
            datasets={
                "merged": {
                    "zarr_path": str(output),
                    "source_type": "materialized_pose_crop",
                    "input_format": "gray",
                    "keypoint_run": result.run_name,
                }
            },
            task="pose",
            target_size=64,
            augmentation_enabled=False,
            model_input_shape_hw=(64, 64),
        ),
        mode="val",
    )
    assert len(loader) == 2
    sample = loader[0]
    assert sample["img"].shape == (3, 64, 64)
    assert sample["keypoints"].shape == (1, 9)
    np.testing.assert_array_equal(sample["keypoints"].reshape(1, 3, 3)[..., 2], 2)
    config_path = tmp_path / "pose.yaml"
    _write_merged_config(
        source_config_path=None,
        out_config=config_path,
        merged_zarr=output,
        dataset_name="materialized_reuse",
        source_type="materialized_pose_crop",
        input_format="gray",
        keypoint_run=result.run_name,
        train_ratio=0.5,
        val_ratio=0.5,
        random_seed=7,
        kpt_shape=(3, 3),
        target_roi_hw=(64, 64),
    )
    assert PoseConfig.from_yaml(config_path).training_params.imgsz == 64
    crop_run = root["crop_runs"][result.run_name]
    crop_run["roi_images"][0, 0, 0] = 99
    with pytest.raises(ValueError, match="logical dataset hash mismatch"):
        validate_merged_keypoint_training_zarr(output, expected_total_samples=4)


def test_materialized_source_refuses_tampered_pixels_and_partial_visibility(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.zarr"
    _source(path, pixel_value=1)
    root = zarr.open_group(str(path), mode="a", use_consolidated=False)
    run = root[f"crop_runs/{RUN_ID}"]
    inspect_materialized_pose_crop_source(run, run_id=RUN_ID)
    run["roi_images"][0, 0, 0] = 9
    with pytest.raises(ValueError, match="digest mismatch"):
        inspect_materialized_pose_crop_source(run, run_id=RUN_ID)
    run["roi_images"][0, 0, 0] = 1
    run["keypoint_visibility"][0, 0] = 0
    run.attrs["keypoint_visibility_sha256"] = _digest(
        np.asarray(run["keypoint_visibility"][:])
    )
    inspect_materialized_pose_crop_source(run, run_id=RUN_ID)
    manifest = _manifest([path])
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="all materialized keypoints visible"):
        _export_merged(
            manifest_payload=manifest,
            manifest_path=manifest_path,
            out_zarr=tmp_path / "merged.zarr",
            merged_dataset_id="partial_merged",
            overwrite=False,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
            seed=7,
            copy_batch_size=2,
            row_gate_policy="auto",
            invocation={},
            roi_transform_mode="strict",
            target_roi_hw=(64, 64),
            split_unit="leakage_group",
            use_storage_contract_v2=True,
        )
