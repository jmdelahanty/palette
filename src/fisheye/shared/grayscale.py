"""Named grayscale conversion conventions used by Palette pixel paths.

The conventions in this module intentionally differ. ``LUMA_BT601_MATLAB`` uses
``0.2989/0.5870/0.1140`` weights, ``LUMA_BT601_CV2`` uses
``0.299/0.587/0.114`` weights, and ``UNWEIGHTED_MEAN`` averages RGB-like
channels without luma weights. Existing callsites choose among these as part of
their pixel contract; changing values or switching a callsite between
conventions changes produced pixels.

See ``docs/diagnostics/pixel_decode_exposure_census_2026-07-02.md`` and
``docs/video_pixel_model_input_contract.md`` before changing a convention or
retargeting a callsite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GrayscaleConvention:
    """Human-readable grayscale convention metadata."""

    name: str
    description: str
    weights: tuple[float, float, float] | None = None


LUMA_BT601_MATLAB = GrayscaleConvention(
    name="rgb_to_gray_bt601_matlab_0_2989_0_5870_0_1140",
    description="BT.601-style RGB luma using the legacy Matlab-style 0.2989/0.5870/0.1140 weights.",
    weights=(0.2989, 0.5870, 0.1140),
)

LUMA_BT601_CV2 = GrayscaleConvention(
    name="rgb_to_gray_bt601_cv2_0_299_0_587_0_114",
    description="BT.601-style RGB luma using the cv2-style 0.299/0.587/0.114 weights.",
    weights=(0.299, 0.587, 0.114),
)

UNWEIGHTED_MEAN = GrayscaleConvention(
    name="rgb_channel_unweighted_mean",
    description="Unweighted arithmetic mean over RGB-like channels.",
)


def rgb_to_gray_bt601_cv2_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB-like arrays to uint8 using the existing cv2-style luma path."""

    arr = np.asarray(rgb)
    if arr.ndim < 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB-like array with at least 3 channels, got shape={arr.shape}")
    weights = LUMA_BT601_CV2.weights
    assert weights is not None
    gray = (
        weights[0] * arr[..., 0].astype(np.float32)
        + weights[1] * arr[..., 1].astype(np.float32)
        + weights[2] * arr[..., 2].astype(np.float32)
    )
    return gray.astype(np.uint8)


def rgb_to_gray_bt601_matlab_torch(
    rgb: Any,
    *,
    weights_dtype: Any,
) -> Any:
    """Return weighted luma using the existing import-video torch expression."""

    import torch

    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB tensor with last dimension 3, got shape={tuple(rgb.shape)}")
    weights = torch.tensor(
        LUMA_BT601_MATLAB.weights,
        device=rgb.device,
        dtype=weights_dtype,
    ).view(*([1] * (rgb.ndim - 1)), 3)
    return (rgb * weights).sum(dim=-1)


def rgb_to_gray_unweighted_mean_torch(
    rgb: Any,
    *,
    accumulator_dtype: Any,
) -> Any:
    """Convert RGB-like tensors to uint8 via the existing unweighted mean path."""

    import torch

    if rgb.ndim < 1:
        raise ValueError(f"Expected RGB-like tensor, got shape={tuple(rgb.shape)}")
    return rgb.to(accumulator_dtype).mean(dim=-1).to(torch.uint8)
