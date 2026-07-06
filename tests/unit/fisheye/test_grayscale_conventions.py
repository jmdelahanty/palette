from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fisheye.capture.import_video import _decode_contract_metadata
from fisheye.shared.crop_image_source import _to_grayscale_uint8
from fisheye.shared.grayscale import (
    LUMA_BT601_MATLAB,
    LUMA_BT601_CV2,
    UNWEIGHTED_MEAN,
    rgb_to_gray_bt601_matlab_torch,
    rgb_to_gray_unweighted_mean_torch,
)
from fisheye.shared.roi_pixel_contract import crop_run_pixel_contract


def test_import_video_matlab_luma_matches_legacy_inline_expression() -> None:
    frames = torch.tensor(
        [
            [
                [[0, 17, 255], [11, 29, 43]],
                [[250, 128, 3], [64, 65, 66]],
            ]
        ],
        dtype=torch.uint8,
    ).to(torch.float32)
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 1, 1, 3)
    expected = ((frames * weights).sum(dim=-1)).clamp(0, 255).to(torch.uint8)

    actual = rgb_to_gray_bt601_matlab_torch(
        frames,
        weights_dtype=torch.float32,
    ).clamp(0, 255).to(torch.uint8)

    assert torch.equal(actual, expected)


def test_import_video_matlab_luma_fp16_matches_legacy_inline_expression() -> None:
    frames = torch.tensor(
        [[[[0, 17, 255], [11, 29, 43]], [[250, 128, 3], [64, 65, 66]]]],
        dtype=torch.uint8,
    ).to(torch.float32)
    work_tensor = frames.half()
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float16).view(1, 1, 1, 3)
    expected = ((work_tensor * weights).sum(dim=-1)).clamp(0, 255).to(torch.uint8)

    actual = rgb_to_gray_bt601_matlab_torch(
        work_tensor,
        weights_dtype=torch.float16,
    ).clamp(0, 255).to(torch.uint8)

    assert torch.equal(actual, expected)


def test_crop_image_source_cv2_luma_matches_legacy_inline_expression() -> None:
    frame = np.array(
        [
            [[0, 17, 255], [11, 29, 43], [250, 128, 3]],
            [[64, 65, 66], [255, 1, 2], [7, 149, 211]],
        ],
        dtype=np.uint8,
    )
    expected = (
        0.299 * frame[..., 0].astype(np.float32)
        + 0.587 * frame[..., 1].astype(np.float32)
        + 0.114 * frame[..., 2].astype(np.float32)
    ).astype(np.uint8)

    actual = _to_grayscale_uint8(frame)

    assert np.array_equal(actual, expected)


def test_tracking_crop_gpu_mean_float32_matches_legacy_inline_expression() -> None:
    frames = torch.tensor(
        [
            [[[0, 17, 255], [11, 29, 43]], [[250, 128, 3], [64, 65, 66]]],
            [[[8, 9, 10], [123, 231, 132]], [[255, 255, 254], [1, 2, 4]]],
        ],
        dtype=torch.uint8,
    )
    expected = frames.to(torch.float32).mean(dim=-1).to(torch.uint8)

    actual = rgb_to_gray_unweighted_mean_torch(frames, accumulator_dtype=torch.float32)

    assert torch.equal(actual, expected)


def test_tracking_crop_gpu_chunk_mean_float16_matches_legacy_inline_expression() -> None:
    frames = torch.tensor(
        [
            [[[0, 17, 255], [11, 29, 43]], [[250, 128, 3], [64, 65, 66]]],
            [[[8, 9, 10], [123, 231, 132]], [[255, 255, 254], [1, 2, 4]]],
        ],
        dtype=torch.uint8,
    )
    expected = frames.to(torch.float16).mean(dim=-1).to(torch.uint8)

    actual = rgb_to_gray_unweighted_mean_torch(frames, accumulator_dtype=torch.float16)

    assert torch.equal(actual, expected)


def test_decode_contract_metadata_records_import_video_convention_name() -> None:
    metadata = _decode_contract_metadata("cuda:0")

    assert metadata["stored_luma_convention"] == LUMA_BT601_MATLAB.name
    assert metadata["stored_luma_weights"] == list(LUMA_BT601_MATLAB.weights or ())


def test_crop_gpu_pixel_contract_records_channel_mean_convention_name() -> None:
    contract = crop_run_pixel_contract(
        crop_storage_mode="materialized",
        video_source_type="external",
        acceleration="gpu",
    )

    assert contract["grayscale_convention"] == UNWEIGHTED_MEAN.name
    assert LUMA_BT601_CV2.name != LUMA_BT601_MATLAB.name
    assert LUMA_BT601_CV2.weights != LUMA_BT601_MATLAB.weights
