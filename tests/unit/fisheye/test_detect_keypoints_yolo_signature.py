from inspect import signature
from typing import Any

import pytest

from fisheye.detection import detect_keypoints_yolo as yolo_mod
from fisheye.detection.detect_keypoints_yolo import (
    DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    detect_keypoints_yolo,
)


class _FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeGroup(dict):
    def __init__(self, children: dict[str, Any] | None = None, *, attrs: dict[str, Any] | None = None) -> None:
        super().__init__(children or {})
        self.attrs = attrs or {}

    def __getitem__(self, key: str) -> Any:
        if "/" not in key:
            return super().__getitem__(key)
        current: Any = self
        for token in key.split("/"):
            current = dict.__getitem__(current, token)
        return current


def test_detect_keypoints_yolo_accepts_mask_threshold() -> None:
    params = signature(detect_keypoints_yolo).parameters
    assert "mask_threshold" in params
    assert "registry" in params
    assert "roi_cache_policy" in params
    assert "roi_cache_dir" in params
    assert "roi_live_acceleration" in params
    assert "roi_live_gpu_chunk_frames" in params
    assert "input_mode" in params
    assert "model_input_transform_mode" in params
    assert "coordinate_contract_mode" in params
    assert params["coordinate_contract_mode"].default == "canonical"
    assert "profile_timings" in params
    assert "keypoint_roi_shard_rows" in params
    assert "keypoint_frame_shard_rows" in params
    assert params["keypoint_roi_shard_rows"].default == DEFAULT_KEYPOINT_ROI_SHARD_ROWS
    assert params["keypoint_frame_shard_rows"].default == DEFAULT_KEYPOINT_FRAME_SHARD_ROWS


def test_keypoint_cli_defaults_to_sharding_and_supports_regular_chunk_opt_out() -> None:
    parser = yolo_mod._build_arg_parser()
    default_args = parser.parse_args(["archive.zarr", "--model", "pose.pt"])
    regular_args = parser.parse_args(
        ["archive.zarr", "--model", "pose.pt", "--no-keypoint-sharding"]
    )

    assert default_args.keypoint_roi_shard_rows == DEFAULT_KEYPOINT_ROI_SHARD_ROWS
    assert default_args.keypoint_frame_shard_rows == DEFAULT_KEYPOINT_FRAME_SHARD_ROWS
    assert default_args.coordinate_contract_mode == "canonical"
    assert regular_args.keypoint_roi_shard_rows is None


def test_keypoint_cli_requires_explicit_legacy_mode_for_collection_shards() -> None:
    parser = yolo_mod._build_arg_parser()
    args = parser.parse_args(
        [
            "archive.zarr",
            "--model",
            "pose.pt",
            "--output-parent",
            "keypoint_shard_runs",
            "--coordinate-contract-mode",
            "legacy_noncanonical",
        ]
    )

    assert args.coordinate_contract_mode == "legacy_noncanonical"


def test_canonical_keypoint_mode_rejects_collection_shard_before_io() -> None:
    with pytest.raises(ValueError, match="Collection shards must explicitly use"):
        detect_keypoints_yolo(
            "missing.zarr",
            "missing.pt",
            output_parent="keypoint_shard_runs",
        )


def test_final_keypoint_parent_rejects_legacy_mode_before_io() -> None:
    with pytest.raises(ValueError, match="Final keypoints_runs are canonical-only"):
        detect_keypoints_yolo(
            "missing.zarr",
            "missing.pt",
            coordinate_contract_mode="legacy_noncanonical",
        )


def test_resolve_full_image_shape_prefers_raw_video_shape() -> None:
    root = _FakeGroup(
        {
            "raw_video": _FakeGroup({"images_full": _FakeArray((7, 60, 50))}),
        },
        attrs={"width": 10, "height": 20},
    )
    crop_group = _FakeGroup(attrs={"width": 4512, "height": 4512})

    shape, total_frames = yolo_mod._resolve_full_image_shape(root, crop_group)

    assert shape == (60, 50)
    assert total_frames == 7


def test_resolve_full_image_shape_uses_crop_run_dimensions_without_raw_video() -> None:
    root = _FakeGroup(attrs={})
    crop_group = _FakeGroup(attrs={"width": 4512, "height": 4512, "total_frames": 143447})

    shape, total_frames = yolo_mod._resolve_full_image_shape(root, crop_group)

    assert shape == (4512, 4512)
    assert total_frames == 143447


def test_resolve_full_image_shape_uses_root_dimension_aliases_without_raw_video() -> None:
    root = _FakeGroup(attrs={"source_video_width": "4512", "source_video_height": 4512, "n_frames": 12})
    crop_group = _FakeGroup(attrs={})

    shape, total_frames = yolo_mod._resolve_full_image_shape(root, crop_group)

    assert shape == (4512, 4512)
    assert total_frames == 12


def test_resolve_full_image_shape_rejects_missing_dimensions_without_raw_video() -> None:
    root = _FakeGroup(attrs={})
    crop_group = _FakeGroup(attrs={})

    with pytest.raises(ValueError, match="crop-run width/height"):
        yolo_mod._resolve_full_image_shape(root, crop_group)
