"""Fixed-size bitpacked storage helpers for binary ROI masks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

MASK_BITPACKED_SCHEMA_ID = "palette_mask_bitpacked_binary_v1"
MASK_BITPACKED_ENCODING = "bitpacked_binary_v1"
MASK_BITPACKED_VALUE_SEMANTICS = "binary_0_1"
MASK_BITPACKED_LAYOUT = "packed_width_array"
MASK_BITPACKED_AXIS = "width"
MASK_BITPACKED_BITORDER = "little"


def packed_width_bytes(width: int) -> int:
    """Return the packed byte width for a binary mask row."""

    width_int = int(width)
    if width_int <= 0:
        raise ValueError(f"Mask width must be positive, got {width!r}.")
    return (width_int + 7) // 8


def normalize_binary_mask_stack(masks: np.ndarray) -> np.ndarray:
    """Normalize ``(N,H,W)`` or ``(N,C,H,W)`` masks to binary ``(N,C,H,W)``."""

    values = np.asarray(masks)
    if values.ndim == 3:
        values = values[:, None, :, :]
    if values.ndim != 4:
        raise ValueError(f"Expected mask stack with shape (N,C,H,W), got {values.shape}.")
    return np.asarray(values > 0, dtype=np.uint8)


def pack_binary_mask_stack(masks: np.ndarray) -> np.ndarray:
    """Pack binary masks along the width axis into uint8 bytes."""

    binary = normalize_binary_mask_stack(masks)
    return np.packbits(binary, axis=-1, bitorder=MASK_BITPACKED_BITORDER)


def unpack_binary_mask_stack(
    packed: np.ndarray,
    *,
    logical_width: int,
) -> np.ndarray:
    """Unpack width-packed binary masks to dense ``uint8`` masks."""

    width = int(logical_width)
    if width <= 0:
        raise ValueError(f"logical_width must be positive, got {logical_width!r}.")
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 4:
        raise ValueError(f"Expected packed mask stack with shape (N,C,H,Wpacked), got {values.shape}.")
    unpacked = np.unpackbits(
        values,
        axis=-1,
        count=width,
        bitorder=MASK_BITPACKED_BITORDER,
    )
    return np.asarray(unpacked > 0, dtype=np.uint8)


def bitpacked_encoded_shape(logical_shape: Sequence[int]) -> tuple[int, int, int, int]:
    """Return ``(N,C,H,packed_width_bytes)`` for a logical mask shape."""

    shape = tuple(int(value) for value in logical_shape)
    if len(shape) == 3:
        n_rows, height, width = shape
        return (n_rows, 1, height, packed_width_bytes(width))
    if len(shape) == 4:
        n_rows, n_channels, height, width = shape
        return (n_rows, n_channels, height, packed_width_bytes(width))
    raise ValueError(f"Expected logical mask shape (N,H,W) or (N,C,H,W), got {shape!r}.")
