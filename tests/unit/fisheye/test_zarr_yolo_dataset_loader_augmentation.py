"""Tests for single-sample detection augmentations in the Zarr loader."""

from pathlib import Path
import sys

import numpy as np
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
