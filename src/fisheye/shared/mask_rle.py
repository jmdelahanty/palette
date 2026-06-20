"""Exact typed-array RLE helpers for binary ROI masks.

The encoding is COCO-style run-length encoding over the Fortran-order
``(height, width)`` mask. Counts alternate background, foreground, background,
... and always start with a background run, so foreground masks beginning at the
first pixel have a leading zero count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

MASK_RLE_SCHEMA_ID = "palette_mask_rle_binary_v1"
MASK_RLE_ENCODING = "coco_rle_fortran_v1"
MASK_RLE_VALUE_SEMANTICS = "binary_0_1"


@dataclass(frozen=True)
class EncodedMaskStack:
    """Flat RLE representation for an ``(N, C, H, W)`` binary mask stack."""

    counts: np.ndarray
    indptr: np.ndarray
    present: np.ndarray
    area_px: np.ndarray
    bbox_xyxy: np.ndarray
    shape_hw: tuple[int, int]


@dataclass(frozen=True)
class EncodedMaskComponent:
    """RLE representation for one semantic component across all rows."""

    component_name: str
    component_index: int
    counts: np.ndarray
    indptr: np.ndarray
    present: np.ndarray
    area_px: np.ndarray
    bbox_xyxy: np.ndarray
    shape_hw: tuple[int, int]


@dataclass(frozen=True)
class EncodedMaskComponentStack:
    """Component-separated RLE representation for an ``(N, C, H, W)`` stack."""

    components: tuple[EncodedMaskComponent, ...]
    shape_hw: tuple[int, int]
    n_rows: int


def _as_binary_2d(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {values.shape}.")
    return np.asarray(values > 0, dtype=np.uint8)


def encode_binary_mask_rle(mask: np.ndarray) -> np.ndarray:
    """Encode one binary mask as Fortran-order alternating RLE counts."""

    binary = _as_binary_2d(mask)
    flat = np.ravel(binary, order="F")
    if flat.size == 0:
        return np.asarray([0], dtype=np.uint32)

    change_indices = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change_indices, [flat.size])).astype(np.int64, copy=False)
    counts = np.diff(boundaries)
    if int(flat[0]) != 0:
        counts = np.concatenate(([0], counts))
    return counts.astype(np.uint32, copy=False)


def decode_binary_mask_rle(counts: Sequence[int] | np.ndarray, shape_hw: Sequence[int]) -> np.ndarray:
    """Decode Fortran-order alternating RLE counts into a dense uint8 mask."""

    if len(shape_hw) != 2:
        raise ValueError(f"shape_hw must contain exactly two dimensions, got {shape_hw!r}.")
    height = int(shape_hw[0])
    width = int(shape_hw[1])
    if height < 0 or width < 0:
        raise ValueError(f"Mask dimensions must be non-negative, got {(height, width)!r}.")

    count_values = np.asarray(counts, dtype=np.int64).reshape(-1)
    if np.any(count_values < 0):
        raise ValueError("RLE counts must be non-negative.")

    total = int(height * width)
    if int(np.sum(count_values, dtype=np.int64)) != total:
        raise ValueError(
            f"RLE counts sum to {int(np.sum(count_values, dtype=np.int64))}, "
            f"expected {total} for shape {(height, width)!r}."
        )

    flat = np.empty((total,), dtype=np.uint8)
    offset = 0
    value = 0
    for count in count_values:
        stop = offset + int(count)
        if stop > offset:
            flat[offset:stop] = value
        offset = stop
        value = 1 - value
    return np.reshape(flat, (height, width), order="F")


def binary_mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return half-open ``(x0, y0, x1, y1)`` bbox for foreground pixels."""

    binary = _as_binary_2d(mask)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _normalize_mask_stack(masks: np.ndarray) -> np.ndarray:
    values = np.asarray(masks)
    if values.ndim == 3:
        values = values[:, None, :, :]
    if values.ndim != 4:
        raise ValueError(f"Expected mask stack with shape (N,C,H,W), got {values.shape}.")
    return np.asarray(values > 0, dtype=np.uint8)


def encode_mask_stack_rle(masks: np.ndarray) -> EncodedMaskStack:
    """Encode an ``(N, C, H, W)`` binary stack into flat typed arrays."""

    binary = _normalize_mask_stack(masks)
    n_rows, n_channels, height, width = (int(v) for v in binary.shape)
    row_channel_count = int(n_rows * n_channels)
    indptr = np.zeros((row_channel_count + 1,), dtype=np.int64)
    present = np.zeros((n_rows, n_channels), dtype=bool)
    area_px = np.zeros((n_rows, n_channels), dtype=np.int32)
    bbox_xyxy = np.zeros((n_rows, n_channels, 4), dtype=np.int32)
    encoded_parts: list[np.ndarray] = []

    cursor = 0
    flat_idx = 0
    for row_idx in range(n_rows):
        for channel_idx in range(n_channels):
            mask = binary[row_idx, channel_idx]
            counts = encode_binary_mask_rle(mask)
            encoded_parts.append(counts)
            cursor += int(counts.size)
            indptr[flat_idx + 1] = cursor
            area = int(np.count_nonzero(mask))
            area_px[row_idx, channel_idx] = area
            present[row_idx, channel_idx] = area > 0
            bbox_xyxy[row_idx, channel_idx] = np.asarray(binary_mask_bbox_xyxy(mask), dtype=np.int32)
            flat_idx += 1

    if encoded_parts:
        counts = np.concatenate(encoded_parts).astype(np.uint32, copy=False)
    else:
        counts = np.zeros((0,), dtype=np.uint32)
    return EncodedMaskStack(
        counts=counts,
        indptr=indptr,
        present=present,
        area_px=area_px,
        bbox_xyxy=bbox_xyxy,
        shape_hw=(height, width),
    )


def _default_component_names(n_channels: int) -> tuple[str, ...]:
    return tuple(f"component_{idx}" for idx in range(int(n_channels)))


def encode_mask_component_rle(
    masks: np.ndarray,
    *,
    component_name: str,
    component_index: int,
) -> EncodedMaskComponent:
    """Encode one component mask stack with shape ``(N,H,W)``."""

    values = np.asarray(masks)
    if values.ndim != 3:
        raise ValueError(f"Expected component mask stack with shape (N,H,W), got {values.shape}.")
    binary = np.asarray(values > 0, dtype=np.uint8)
    n_rows, height, width = (int(v) for v in binary.shape)
    indptr = np.zeros((n_rows + 1,), dtype=np.int64)
    present = np.zeros((n_rows,), dtype=bool)
    area_px = np.zeros((n_rows,), dtype=np.int32)
    bbox_xyxy = np.zeros((n_rows, 4), dtype=np.int32)
    encoded_parts: list[np.ndarray] = []
    cursor = 0
    for row_idx in range(n_rows):
        mask = binary[row_idx]
        counts = encode_binary_mask_rle(mask)
        encoded_parts.append(counts)
        cursor += int(counts.size)
        indptr[row_idx + 1] = cursor
        area = int(np.count_nonzero(mask))
        area_px[row_idx] = area
        present[row_idx] = area > 0
        bbox_xyxy[row_idx] = np.asarray(binary_mask_bbox_xyxy(mask), dtype=np.int32)
    counts = np.concatenate(encoded_parts).astype(np.uint32, copy=False) if encoded_parts else np.zeros((0,), dtype=np.uint32)
    return EncodedMaskComponent(
        component_name=str(component_name),
        component_index=int(component_index),
        counts=counts,
        indptr=indptr,
        present=present,
        area_px=area_px,
        bbox_xyxy=bbox_xyxy,
        shape_hw=(height, width),
    )


def encode_mask_component_stack_rle(
    masks: np.ndarray,
    *,
    component_names: Sequence[str] | None = None,
) -> EncodedMaskComponentStack:
    """Encode ``(N,C,H,W)`` masks into per-component RLE payloads."""

    binary = _normalize_mask_stack(masks)
    n_rows, n_channels, height, width = (int(v) for v in binary.shape)
    names = tuple(str(value) for value in component_names) if component_names is not None else _default_component_names(n_channels)
    if len(names) != n_channels:
        raise ValueError(f"Expected {n_channels} component names, got {len(names)}.")
    components = tuple(
        encode_mask_component_rle(
            binary[:, channel_idx],
            component_name=names[channel_idx],
            component_index=channel_idx,
        )
        for channel_idx in range(n_channels)
    )
    return EncodedMaskComponentStack(
        components=components,
        shape_hw=(height, width),
        n_rows=n_rows,
    )


def concatenate_encoded_mask_components(shards: Sequence[EncodedMaskComponent]) -> EncodedMaskComponent:
    """Concatenate row-sharded RLE payloads for one component."""

    shard_list = tuple(shards)
    if not shard_list:
        raise ValueError("Cannot concatenate zero component shards without component metadata.")
    first = shard_list[0]
    shape_hw = tuple(int(v) for v in first.shape_hw)
    component_name = str(first.component_name)
    component_index = int(first.component_index)
    counts_parts: list[np.ndarray] = []
    indptr_parts: list[np.ndarray] = [np.asarray([0], dtype=np.int64)]
    present_parts: list[np.ndarray] = []
    area_parts: list[np.ndarray] = []
    bbox_parts: list[np.ndarray] = []
    cursor = 0
    for shard in shard_list:
        if tuple(int(v) for v in shard.shape_hw) != shape_hw:
            raise ValueError(
                f"Cannot concatenate component RLE shards with different shapes: "
                f"{shape_hw!r} vs {shard.shape_hw!r}."
            )
        if str(shard.component_name) != component_name or int(shard.component_index) != component_index:
            raise ValueError("Cannot concatenate component RLE shards with different component identities.")
        if int(shard.indptr[0]) != 0 or int(shard.indptr[-1]) != int(shard.counts.size):
            raise ValueError("Component shard indptr must be local to the shard counts payload.")
        counts_parts.append(np.asarray(shard.counts, dtype=np.uint32))
        indptr_parts.append(np.asarray(shard.indptr[1:], dtype=np.int64) + int(cursor))
        cursor += int(shard.counts.size)
        present_parts.append(np.asarray(shard.present, dtype=bool))
        area_parts.append(np.asarray(shard.area_px, dtype=np.int32))
        bbox_parts.append(np.asarray(shard.bbox_xyxy, dtype=np.int32))
    return EncodedMaskComponent(
        component_name=component_name,
        component_index=component_index,
        counts=np.concatenate(counts_parts).astype(np.uint32, copy=False),
        indptr=np.concatenate(indptr_parts).astype(np.int64, copy=False),
        present=np.concatenate(present_parts, axis=0).astype(bool, copy=False),
        area_px=np.concatenate(area_parts, axis=0).astype(np.int32, copy=False),
        bbox_xyxy=np.concatenate(bbox_parts, axis=0).astype(np.int32, copy=False),
        shape_hw=shape_hw,
    )


def concatenate_encoded_mask_component_stacks(
    shards: Sequence[EncodedMaskComponentStack],
) -> EncodedMaskComponentStack:
    """Concatenate row-sharded component-separated RLE stacks."""

    shard_list = tuple(shards)
    if not shard_list:
        return EncodedMaskComponentStack(components=(), shape_hw=(0, 0), n_rows=0)
    first = shard_list[0]
    shape_hw = tuple(int(v) for v in first.shape_hw)
    component_count = len(first.components)
    component_identities = tuple((component.component_name, component.component_index) for component in first.components)
    combined: list[EncodedMaskComponent] = []
    for shard in shard_list:
        if tuple(int(v) for v in shard.shape_hw) != shape_hw:
            raise ValueError(f"Cannot concatenate component stacks with different shapes: {shape_hw!r} vs {shard.shape_hw!r}.")
        identities = tuple((component.component_name, component.component_index) for component in shard.components)
        if len(shard.components) != component_count or identities != component_identities:
            raise ValueError("Cannot concatenate component stacks with different component identities.")
    for component_idx in range(component_count):
        combined.append(concatenate_encoded_mask_components([shard.components[component_idx] for shard in shard_list]))
    return EncodedMaskComponentStack(
        components=tuple(combined),
        shape_hw=shape_hw,
        n_rows=sum(int(shard.n_rows) for shard in shard_list),
    )


def component_stack_from_flat_rle(
    flat: EncodedMaskStack,
    *,
    component_names: Sequence[str] | None = None,
) -> EncodedMaskComponentStack:
    """Convert row-major flat ``(row, channel)`` RLE into component groups."""

    if flat.present.ndim != 2:
        raise ValueError(f"Flat present array must be 2D, got shape {flat.present.shape!r}.")
    n_rows, n_channels = (int(v) for v in flat.present.shape)
    names = tuple(str(value) for value in component_names) if component_names is not None else _default_component_names(n_channels)
    if len(names) != n_channels:
        raise ValueError(f"Expected {n_channels} component names, got {len(names)}.")

    components: list[EncodedMaskComponent] = []
    for channel_idx, component_name in enumerate(names):
        counts_parts: list[np.ndarray] = []
        indptr = np.zeros((n_rows + 1,), dtype=np.int64)
        cursor = 0
        for row_idx in range(n_rows):
            flat_idx = row_idx * n_channels + channel_idx
            start = int(flat.indptr[flat_idx])
            stop = int(flat.indptr[flat_idx + 1])
            part = np.asarray(flat.counts[start:stop], dtype=np.uint32)
            counts_parts.append(part)
            cursor += int(part.size)
            indptr[row_idx + 1] = cursor
        counts = np.concatenate(counts_parts).astype(np.uint32, copy=False) if counts_parts else np.zeros((0,), dtype=np.uint32)
        components.append(
            EncodedMaskComponent(
                component_name=str(component_name),
                component_index=int(channel_idx),
                counts=counts,
                indptr=indptr,
                present=np.asarray(flat.present[:, channel_idx], dtype=bool),
                area_px=np.asarray(flat.area_px[:, channel_idx], dtype=np.int32),
                bbox_xyxy=np.asarray(flat.bbox_xyxy[:, channel_idx, :], dtype=np.int32),
                shape_hw=tuple(int(v) for v in flat.shape_hw),
            )
        )
    return EncodedMaskComponentStack(
        components=tuple(components),
        shape_hw=tuple(int(v) for v in flat.shape_hw),
        n_rows=n_rows,
    )


def concatenate_encoded_mask_stacks(shards: Sequence[EncodedMaskStack]) -> EncodedMaskStack:
    """Concatenate row-sharded RLE stacks into one flat representation.

    Each shard must contain the same channel count and ROI shape. ``indptr`` is
    offset-adjusted so the returned ``counts`` array can be written as one
    canonical payload.
    """

    shard_list = tuple(shards)
    if not shard_list:
        return EncodedMaskStack(
            counts=np.zeros((0,), dtype=np.uint32),
            indptr=np.asarray([0], dtype=np.int64),
            present=np.zeros((0, 0), dtype=bool),
            area_px=np.zeros((0, 0), dtype=np.int32),
            bbox_xyxy=np.zeros((0, 0, 4), dtype=np.int32),
            shape_hw=(0, 0),
        )

    shape_hw = tuple(int(v) for v in shard_list[0].shape_hw)
    channel_count = int(shard_list[0].present.shape[1]) if shard_list[0].present.ndim == 2 else 0
    counts_parts: list[np.ndarray] = []
    indptr_parts: list[np.ndarray] = [np.asarray([0], dtype=np.int64)]
    present_parts: list[np.ndarray] = []
    area_parts: list[np.ndarray] = []
    bbox_parts: list[np.ndarray] = []
    cursor = 0
    for shard in shard_list:
        if tuple(int(v) for v in shard.shape_hw) != shape_hw:
            raise ValueError(f"Cannot concatenate RLE shards with different shapes: {shape_hw!r} vs {shard.shape_hw!r}.")
        if shard.present.ndim != 2:
            raise ValueError(f"Shard present array must be 2D, got shape {shard.present.shape!r}.")
        if int(shard.present.shape[1]) != channel_count:
            raise ValueError(
                f"Cannot concatenate RLE shards with different channel counts: "
                f"{channel_count} vs {int(shard.present.shape[1])}."
            )
        if int(shard.indptr[0]) != 0 or int(shard.indptr[-1]) != int(shard.counts.size):
            raise ValueError("Shard indptr must be local to the shard counts payload.")
        counts_parts.append(np.asarray(shard.counts, dtype=np.uint32))
        indptr_parts.append(np.asarray(shard.indptr[1:], dtype=np.int64) + int(cursor))
        cursor += int(shard.counts.size)
        present_parts.append(np.asarray(shard.present, dtype=bool))
        area_parts.append(np.asarray(shard.area_px, dtype=np.int32))
        bbox_parts.append(np.asarray(shard.bbox_xyxy, dtype=np.int32))

    return EncodedMaskStack(
        counts=np.concatenate(counts_parts).astype(np.uint32, copy=False) if counts_parts else np.zeros((0,), dtype=np.uint32),
        indptr=np.concatenate(indptr_parts).astype(np.int64, copy=False),
        present=np.concatenate(present_parts, axis=0).astype(bool, copy=False),
        area_px=np.concatenate(area_parts, axis=0).astype(np.int32, copy=False),
        bbox_xyxy=np.concatenate(bbox_parts, axis=0).astype(np.int32, copy=False),
        shape_hw=shape_hw,
    )


def decode_mask_stack_rle(
    counts: np.ndarray,
    indptr: np.ndarray,
    *,
    n_rows: int,
    n_channels: int,
    shape_hw: Sequence[int],
) -> np.ndarray:
    """Decode flat typed-array RLE back to ``(N, C, H, W)`` uint8 masks."""

    row_count = int(n_rows)
    channel_count = int(n_channels)
    if row_count < 0 or channel_count < 0:
        raise ValueError("n_rows and n_channels must be non-negative.")
    pointers = np.asarray(indptr, dtype=np.int64).reshape(-1)
    expected = row_count * channel_count + 1
    if pointers.size != expected:
        raise ValueError(f"indptr has {pointers.size} entries; expected {expected}.")
    if pointers.size and (int(pointers[0]) != 0 or np.any(np.diff(pointers) < 0)):
        raise ValueError("indptr must start at zero and be monotonically non-decreasing.")
    count_values = np.asarray(counts, dtype=np.uint32).reshape(-1)
    if pointers.size and int(pointers[-1]) != int(count_values.size):
        raise ValueError(f"indptr terminates at {int(pointers[-1])}, but counts has {count_values.size} values.")

    height = int(shape_hw[0])
    width = int(shape_hw[1])
    output = np.zeros((row_count, channel_count, height, width), dtype=np.uint8)
    flat_idx = 0
    for row_idx in range(row_count):
        for channel_idx in range(channel_count):
            start = int(pointers[flat_idx])
            stop = int(pointers[flat_idx + 1])
            output[row_idx, channel_idx] = decode_binary_mask_rle(count_values[start:stop], (height, width))
            flat_idx += 1
    return output


def decode_mask_component_rle(component: EncodedMaskComponent) -> np.ndarray:
    """Decode one component RLE payload to ``(N,H,W)``."""

    n_rows = int(component.present.shape[0])
    height, width = (int(v) for v in component.shape_hw)
    output = np.zeros((n_rows, height, width), dtype=np.uint8)
    for row_idx in range(n_rows):
        start = int(component.indptr[row_idx])
        stop = int(component.indptr[row_idx + 1])
        output[row_idx] = decode_binary_mask_rle(component.counts[start:stop], component.shape_hw)
    return output


def decode_mask_component_stack_rle(stack: EncodedMaskComponentStack) -> np.ndarray:
    """Decode component-separated RLE to ``(N,C,H,W)``."""

    n_rows = int(stack.n_rows)
    n_channels = len(stack.components)
    height, width = (int(v) for v in stack.shape_hw)
    output = np.zeros((n_rows, n_channels, height, width), dtype=np.uint8)
    for component in stack.components:
        output[:, int(component.component_index)] = decode_mask_component_rle(component)
    return output
