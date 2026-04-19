"""Tests for single-sample detection augmentations in the Zarr loader."""

from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.zarr_yolo_dataset_loader import ZarrDatasetConfig, create_zarr_dataset


def _write_detect_zarr(path: Path, *, num_frames: int = 2, frame_chunk: int = 1) -> None:
    root = zarr.open_group(str(path), mode="w")
    raw = root.create_group("raw_video")

    row = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    images = []
    for idx in range(num_frames):
        images.append(np.rot90(row, k=idx % 4))
    raw.create_array(
        "images_ds",
        data=np.stack(images, axis=0),
        chunks=(max(1, frame_chunk), 32, 32),
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_test"
    crop = crop_parent.create_group("crop_test")
    crop.attrs["detection_source_type"] = "manual"
    crop.attrs["includes_interpolated"] = False
    crop.create_array(
        "roi_images",
        data=np.stack(images, axis=0),
        chunks=(max(1, frame_chunk), 32, 32),
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.stack(
            [
                np.array(
                    [0.2 + 0.6 * (idx / max(1, num_frames - 1)), 0.5, 0.4, 0.4],
                    dtype=np.float32,
                )
                for idx in range(num_frames)
            ],
            axis=0,
        ),
        chunks=(max(1, min(num_frames, 8)), 4),
    )
    crop.create_array(
        "frame_indices",
        data=np.arange(num_frames, dtype=np.int64),
        chunks=(max(1, min(num_frames, 8)),),
    )
    crop.create_array(
        "detection_source",
        data=np.zeros((num_frames,), dtype=np.int8),
        chunks=(max(1, min(num_frames, 8)),),
    )


def _write_curated_refined_detect_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((2, 32, 32), dtype=np.uint8),
        chunks=(1, 32, 32),
    )
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.create_array("refined_row_ids", data=np.array([0, 1], dtype=np.int64), chunks=(2,))
    refined.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), chunks=(2,))
    refined.create_array("entity_ids", data=np.array([0, 0], dtype=np.int32), chunks=(2,))
    refined.create_array(
        "bbox_img_xyxy",
        data=np.array([[2.0, 2.0, 8.0, 8.0], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float64),
        chunks=(2, 4),
    )
    refined.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.2, 0.2], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float64),
        chunks=(2, 4),
    )
    refined.create_array("status_codes", data=np.array([0, 2], dtype=np.int8), chunks=(2,))
    refined.create_array("source_kind_codes", data=np.array([1, 0], dtype=np.int8), chunks=(2,))
    refined.create_array("review_state_codes", data=np.array([1, 1], dtype=np.int8), chunks=(2,))
    refined.create_array("keypoints_state_codes", data=np.array([0, 0], dtype=np.int8), chunks=(2,))
    refined.create_array("subject_mask_state_codes", data=np.array([0, 0], dtype=np.int8), chunks=(2,))
    refined.create_array("eye_mask_state_codes", data=np.array([0, 0], dtype=np.int8), chunks=(2,))
    refined.create_array("swim_bladder_state_codes", data=np.array([0, 0], dtype=np.int8), chunks=(2,))
    refined.create_array("detection_source", data=np.array([0, 0], dtype=np.int8), chunks=(2,))


def _write_pose_merged_zarr_with_box_only(
    path: Path,
    *,
    keypoint_labels: tuple[str, ...] = ("swim_bladder", "eye_left", "eye_right"),
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_purpose"] = "training"

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "merged_export_test"
    crop = crop_parent.create_group("merged_export_test")
    crop.attrs["detection_source_type"] = "filtered"
    crop.attrs["includes_interpolated"] = False
    crop.create_array(
        "roi_images",
        data=np.zeros((3, 32, 32), dtype=np.uint8),
        chunks=(1, 32, 32),
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.array(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.4, 0.6, 0.3, 0.2],
                [0.6, 0.4, 0.2, 0.3],
            ],
            dtype=np.float32,
        ),
        chunks=(3, 4),
    )
    crop.create_array(
        "frame_indices",
        data=np.arange(3, dtype=np.int64),
        chunks=(3,),
    )
    crop.create_array(
        "detection_source",
        data=np.zeros((3,), dtype=np.int8),
        chunks=(3,),
    )

    kp_parent = root.create_group("keypoints_runs")
    kp_parent.attrs["latest"] = "merged_export_test"
    kp = kp_parent.create_group("merged_export_test")
    kp.attrs["method"] = "merged_export"
    kp.attrs["row_gate_applied"] = True
    kp.attrs["source_crop_run"] = "merged_export_test"
    kp.attrs["keypoint_labels"] = list(keypoint_labels)
    kp.attrs["pose_schema"] = {
        "name": "test_pose_schema",
        "keypoint_labels": list(keypoint_labels),
        "nodes": [{"id": idx, "name": label} for idx, label in enumerate(keypoint_labels)],
        "edges": [[0, 1], [0, 2], [1, 2]],
    }
    kp.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[10.0, 10.0], [12.0, 12.0], [14.0, 14.0]],
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                [[8.0, 8.0], [9.0, 9.0], [10.0, 10.0]],
            ],
            dtype=np.float32,
        ),
        chunks=(3, 3, 2),
    )
    kp.create_array(
        "detection_success",
        data=np.array([True, False, True], dtype=np.bool_),
        chunks=(3,),
    )
    kp.create_array(
        "keypoint_box_only",
        data=np.array([False, True, False], dtype=np.bool_),
        chunks=(3,),
    )


def _build_config(
    zarr_path: Path,
    augmentation: dict | None = None,
    *,
    split_train: float = 0.5,
    split_val: float = 0.5,
    chunk_cache_size: int = 0,
) -> ZarrDatasetConfig:
    return ZarrDatasetConfig(
        datasets={
            "sample": {
                "zarr_path": str(zarr_path),
                "source_type": "manual",
                "input_format": "gray",
                "split": {"train": split_train, "val": split_val},
            }
        },
        task="detect",
        random_seed=7,
        sampling_strategy="proportional",
        chunk_cache_size=chunk_cache_size,
        augmentation=augmentation or {},
    )


def test_detect_train_fliplr_applies_to_image_and_bbox(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _write_detect_zarr(zarr_path)

    ds_base = create_zarr_dataset(_build_config(zarr_path), mode="train")
    ds_flip = create_zarr_dataset(_build_config(zarr_path, {"fliplr": 1.0}), mode="train")

    base = ds_base[0]
    flipped = ds_flip[0]
    base_img = base["img"].transpose(1, 2, 0)
    flip_img = flipped["img"].transpose(1, 2, 0)

    assert np.array_equal(flip_img, np.fliplr(base_img))
    assert np.allclose(flipped["bboxes"][:, 0], 1.0 - base["bboxes"][:, 0], atol=1e-6)
    assert np.allclose(flipped["bboxes"][:, 1:], base["bboxes"][:, 1:], atol=1e-6)


def test_detect_val_mode_skips_train_augmentations(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _write_detect_zarr(zarr_path)

    ds_base = create_zarr_dataset(_build_config(zarr_path), mode="val")
    ds_aug = create_zarr_dataset(_build_config(zarr_path, {"fliplr": 1.0, "flipud": 1.0}), mode="val")

    base = ds_base[0]
    aug = ds_aug[0]
    assert np.array_equal(base["img"], aug["img"])
    assert np.array_equal(base["bboxes"], aug["bboxes"])


def test_detect_train_erasing_changes_pixels(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _write_detect_zarr(zarr_path)

    ds_base = create_zarr_dataset(_build_config(zarr_path), mode="train")
    ds_erase = create_zarr_dataset(_build_config(zarr_path, {"erasing": 1.0}), mode="train")

    base = ds_base[0]["img"]
    erased = ds_erase[0]["img"]
    assert not np.array_equal(base, erased)


def test_detect_chunk_cache_reuses_recent_chunk(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _write_detect_zarr(zarr_path, num_frames=4, frame_chunk=2)

    ds = create_zarr_dataset(
        _build_config(zarr_path, split_train=0.75, split_val=0.25, chunk_cache_size=1),
        mode="train",
    )
    assert len(ds.indices) >= 2

    chunk_len = ds.detect_frame_chunk_len[str(zarr_path)]
    chunk_to_positions: dict[int, list[int]] = {}
    for pos, (_, det_idx) in enumerate(ds.indices):
        frame_idx = int(ds.frame_index_cache[str(zarr_path)][det_idx])
        chunk_id = frame_idx // chunk_len
        chunk_to_positions.setdefault(chunk_id, []).append(pos)

    reusable_positions = next(positions for positions in chunk_to_positions.values() if len(positions) >= 2)
    first_pos, second_pos = reusable_positions[0], reusable_positions[1]

    _ = ds[first_pos]
    first_stats = ds.get_chunk_cache_stats()
    _ = ds[second_pos]
    second_stats = ds.get_chunk_cache_stats()

    assert first_stats["chunk_cache_misses"] >= 1
    assert second_stats["chunk_cache_hits"] >= first_stats["chunk_cache_hits"] + 1


def test_pose_loader_supports_box_only_rows_with_visibility_zero(tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_box_only.zarr"
    _write_pose_merged_zarr_with_box_only(zarr_path)

    cfg = ZarrDatasetConfig(
        datasets={
            "pose": {
                "zarr_path": str(zarr_path),
                "source_type": "filtered",
                "input_format": "gray",
                "split": {"train": 1.0, "val": 0.0},
            }
        },
        task="pose",
        random_seed=11,
        sampling_strategy="proportional",
    )
    ds = create_zarr_dataset(cfg, mode="train")
    assert len(ds) == 3

    box_only_pos = next(i for i, (_path, det_idx) in enumerate(ds.indices) if int(det_idx) == 1)
    sample = ds[box_only_pos]
    assert sample["cls"].shape == (1,)
    assert sample["bboxes"].shape == (1, 4)
    assert np.allclose(sample["bboxes"][0], np.array([0.4, 0.6, 0.3, 0.2], dtype=np.float32))
    assert not np.isnan(sample["keypoints"]).any()
    vis = sample["keypoints"].reshape(1, 3, 3)[0, :, 2]
    assert np.allclose(vis, np.zeros((3,), dtype=np.float32))

    full_pos = next(i for i, (_path, det_idx) in enumerate(ds.indices) if int(det_idx) == 0)
    full_sample = ds[full_pos]
    full_vis = full_sample["keypoints"].reshape(1, 3, 3)[0, :, 2]
    assert np.allclose(full_vis, np.full((3,), 2.0, dtype=np.float32))


def test_pose_loader_uses_metadata_keypoint_labels(tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_labels.zarr"
    _write_pose_merged_zarr_with_box_only(
        zarr_path,
        keypoint_labels=("left_eye", "tail_tip", "bladder"),
    )

    cfg = ZarrDatasetConfig(
        datasets={
            "pose": {
                "zarr_path": str(zarr_path),
                "source_type": "filtered",
                "input_format": "gray",
                "split": {"train": 1.0, "val": 0.0},
            }
        },
        task="pose",
        random_seed=11,
        sampling_strategy="proportional",
    )
    ds = create_zarr_dataset(cfg, mode="train")

    assert ds.keypoint_labels == ["eye_left", "tail_tip", "swim_bladder"]


def test_pose_loader_rejects_mixed_keypoint_labels(tmp_path: Path) -> None:
    zarr_a = tmp_path / "pose_a.zarr"
    zarr_b = tmp_path / "pose_b.zarr"
    _write_pose_merged_zarr_with_box_only(
        zarr_a,
        keypoint_labels=("swim_bladder", "eye_left", "eye_right"),
    )
    _write_pose_merged_zarr_with_box_only(
        zarr_b,
        keypoint_labels=("tail_tip", "eye_left", "eye_right"),
    )

    cfg = ZarrDatasetConfig(
        datasets={
            "pose_a": {
                "zarr_path": str(zarr_a),
                "source_type": "filtered",
                "input_format": "gray",
                "split": {"train": 1.0, "val": 0.0},
            },
            "pose_b": {
                "zarr_path": str(zarr_b),
                "source_type": "filtered",
                "input_format": "gray",
                "split": {"train": 1.0, "val": 0.0},
            },
        },
        task="pose",
        random_seed=11,
        sampling_strategy="proportional",
    )

    with pytest.raises(ValueError, match="Mixed keypoint_labels across configured pose datasets"):
        create_zarr_dataset(cfg, mode="train")


def test_detect_loader_supports_curated_refined_root_source(tmp_path: Path) -> None:
    zarr_path = tmp_path / "refined_source.zarr"
    _write_curated_refined_detect_zarr(zarr_path)

    cfg = ZarrDatasetConfig(
        datasets={
            "sample": {
                "zarr_path": str(zarr_path),
                "source_type": "refined",
                "input_format": "gray",
                "split": {"train": 1.0, "val": 0.0},
            }
        },
        task="detect",
        random_seed=3,
        sampling_strategy="proportional",
    )
    ds = create_zarr_dataset(cfg, mode="train")

    assert len(ds) == 1
    sample = ds[0]
    assert sample["bboxes"].shape == (1, 4)
    assert np.allclose(sample["bboxes"][0], np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32))
