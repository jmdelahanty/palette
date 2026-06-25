"""Reversible model-input transforms for native ROI crops.

Prediction runs should store outputs in the native crop coordinate system even
when a model expects a different input size. This module keeps that transform
explicit and invertible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MODEL_INPUT_TRANSFORM_CHOICES = ("auto", "identity", "pad_to_size")


@dataclass(frozen=True)
class ModelInputTransform:
    """Describe a reversible transform from native ROI pixels to model input."""

    name: str
    native_height: int
    native_width: int
    model_height: int
    model_width: int
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0

    @property
    def is_identity(self) -> bool:
        return (
            self.name == "identity"
            and self.native_height == self.model_height
            and self.native_width == self.model_width
            and self.pad_top == 0
            and self.pad_bottom == 0
            and self.pad_left == 0
            and self.pad_right == 0
        )

    @property
    def native_shape(self) -> tuple[int, int]:
        return (self.native_height, self.native_width)

    @property
    def model_shape(self) -> tuple[int, int]:
        return (self.model_height, self.model_width)

    def to_attrs(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "native_shape_hw": [int(self.native_height), int(self.native_width)],
            "model_shape_hw": [int(self.model_height), int(self.model_width)],
            "pad_top": int(self.pad_top),
            "pad_bottom": int(self.pad_bottom),
            "pad_left": int(self.pad_left),
            "pad_right": int(self.pad_right),
            "coordinate_mapping": "native_xy = model_xy - [pad_left, pad_top]",
        }

    def apply_numpy_luma_batch(self, batch: np.ndarray) -> np.ndarray:
        """Apply the transform to an ``(N,H,W)`` luma batch."""
        arr = np.asarray(batch)
        if arr.ndim != 3:
            raise ValueError(f"Expected ROI batch shape (N,H,W), got {arr.shape}")
        if tuple(arr.shape[1:]) != self.native_shape:
            raise ValueError(
                f"ROI batch shape {tuple(arr.shape[1:])} does not match transform native shape {self.native_shape}"
            )
        if self.is_identity:
            return arr
        return np.pad(
            arr,
            ((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)),
            mode="constant",
            constant_values=0,
        )

    def apply_torch_image_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply the transform to an ``(N,C,H,W)`` tensor batch."""
        if batch.ndim != 4:
            raise ValueError(f"Expected tensor batch shape (N,C,H,W), got {tuple(batch.shape)}")
        if tuple(batch.shape[-2:]) != self.native_shape:
            raise ValueError(
                f"Tensor batch shape {tuple(batch.shape[-2:])} does not match transform native shape {self.native_shape}"
            )
        if self.is_identity:
            return batch
        return F.pad(batch, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom), mode="constant", value=0.0)

    def crop_torch_output(self, output: torch.Tensor) -> torch.Tensor:
        """Crop model-space output back to native ROI dimensions."""
        if output.ndim < 2:
            raise ValueError(f"Expected image-like tensor output, got {tuple(output.shape)}")
        if self.is_identity:
            return output
        if tuple(output.shape[-2:]) != self.model_shape:
            raise ValueError(
                f"Model output shape {tuple(output.shape[-2:])} does not match transform model shape {self.model_shape}"
            )
        y0 = self.pad_top
        y1 = y0 + self.native_height
        x0 = self.pad_left
        x1 = x0 + self.native_width
        return output[..., y0:y1, x0:x1]

    def invert_points_xy(self, points_xy: np.ndarray) -> np.ndarray:
        """Map model-input ``xy`` points back to native ROI coordinates."""
        out = np.asarray(points_xy, dtype=np.float64).copy()
        out[..., 0] -= float(self.pad_left)
        out[..., 1] -= float(self.pad_top)
        return out

    def invert_boxes_xyxy(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Map model-input ``xyxy`` boxes back to native ROI coordinates."""
        out = np.asarray(boxes_xyxy, dtype=np.float32).copy()
        if out.shape[-1] != 4:
            raise ValueError(f"Expected xyxy boxes with last dimension 4, got {out.shape}")
        out[..., 0] -= float(self.pad_left)
        out[..., 2] -= float(self.pad_left)
        out[..., 1] -= float(self.pad_top)
        out[..., 3] -= float(self.pad_top)
        return out


def resolve_model_input_transform(
    native_hw: tuple[int, int],
    *,
    mode: str = "auto",
    model_hw: Optional[tuple[int, int]] = None,
) -> ModelInputTransform:
    """Resolve a native-ROI to model-input transform.

    ``auto`` is identity when shapes match and centered zero-padding when the
    requested model shape is larger than the native shape. It intentionally does
    not resize pixels.
    """
    if mode not in MODEL_INPUT_TRANSFORM_CHOICES:
        raise ValueError(f"Unsupported model input transform {mode!r}; expected one of {MODEL_INPUT_TRANSFORM_CHOICES}")
    native_height, native_width = (int(native_hw[0]), int(native_hw[1]))
    if native_height <= 0 or native_width <= 0:
        raise ValueError(f"Native ROI shape must be positive, got {native_hw}")
    if model_hw is None:
        model_height, model_width = native_height, native_width
    else:
        model_height, model_width = (int(model_hw[0]), int(model_hw[1]))
    if model_height <= 0 or model_width <= 0:
        raise ValueError(f"Model input shape must be positive, got {model_hw}")

    if mode == "identity" or (mode == "auto" and (model_height, model_width) == (native_height, native_width)):
        if (model_height, model_width) != (native_height, native_width):
            raise ValueError(
                "identity model input transform requires model shape to match native ROI shape "
                f"({native_height}, {native_width}), got ({model_height}, {model_width})"
            )
        return ModelInputTransform(
            name="identity",
            native_height=native_height,
            native_width=native_width,
            model_height=model_height,
            model_width=model_width,
        )

    if mode in ("auto", "pad_to_size"):
        if model_height < native_height or model_width < native_width:
            raise ValueError(
                "pad_to_size model input transform cannot shrink ROIs; "
                f"native=({native_height}, {native_width}) model=({model_height}, {model_width})"
            )
        pad_y = model_height - native_height
        pad_x = model_width - native_width
        pad_top = pad_y // 2
        pad_left = pad_x // 2
        return ModelInputTransform(
            name="pad_to_size",
            native_height=native_height,
            native_width=native_width,
            model_height=model_height,
            model_width=model_width,
            pad_top=pad_top,
            pad_bottom=pad_y - pad_top,
            pad_left=pad_left,
            pad_right=pad_x - pad_left,
        )

    raise AssertionError(f"Unhandled model input transform mode: {mode}")
