"""Reusable binary-mask geometry helpers.

These functions are deliberately policy-free. They provide small OpenCV/NumPy
primitives used by subject-mask and eye-mask refinement code while leaving
stage-specific review routing in the caller.
"""

from __future__ import annotations

import cv2
import numpy as np

_COORD_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background holes in a 2D binary mask."""

    mask_bool = mask.astype(bool, copy=False)
    if not np.any(mask_bool):
        return mask_bool.copy()
    holes = hole_mask(mask_bool)
    if not np.any(holes):
        return mask_bool.copy()
    return mask_bool | holes


def hole_mask(mask: np.ndarray) -> np.ndarray:
    """Return enclosed background pixels using a cv2 flood-fill from outside.

    The padded border makes the outside background explicit even when the
    top-left pixel in the original mask is foreground.
    """

    mask_bool = mask.astype(bool, copy=False)
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask_bool.shape}")
    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=bool)
    height, width = mask_bool.shape
    padded = np.pad(
        mask_bool.astype(np.uint8, copy=False) * np.uint8(2),
        1,
        mode="constant",
        constant_values=0,
    )
    flood_mask = np.zeros((int(height) + 4, int(width) + 4), dtype=np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 1)
    return padded[1:-1, 1:-1] == 0


def hole_stats(mask: np.ndarray) -> tuple[int, float, int]:
    """Return hole count, filled-area fraction, and hole area for a 2D mask."""

    mask_bool = mask.astype(bool, copy=False)
    if not np.any(mask_bool):
        return 0, 0.0, 0
    holes = hole_mask(mask_bool)
    hole_area = int(np.count_nonzero(holes))
    if hole_area == 0:
        return 0, 0.0, 0
    _labels, count = connected_component_labels(holes)
    denom = max(1, int(np.count_nonzero(mask_bool)) + hole_area)
    return int(count), float(hole_area / denom), hole_area


def connected_component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected foreground components in a 2D binary mask."""

    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32, copy=False), int(num_labels - 1)


def select_component_near_point(mask: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    """Return the connected component whose centroid is nearest ``point_xy``.

    If the foreground has exactly one component, the binary mask is returned
    immediately. That fast path preserves the legacy nearest-centroid result
    while avoiding unnecessary stats work for the common case.
    """

    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    selected = np.zeros_like(binary, dtype=bool)
    if int(np.count_nonzero(binary)) <= 0:
        return selected

    label_count, labels = cv2.connectedComponents(binary, connectivity=8)
    if label_count <= 1:
        return selected
    if label_count == 2:
        return binary.astype(bool, copy=False)

    point = np.asarray(point_xy, dtype=np.float32).reshape(-1)
    x_grid, y_grid = coordinate_grids(binary.shape)
    foreground = binary > 0
    label_values = labels[foreground].reshape(-1)
    areas = np.bincount(label_values, minlength=int(label_count)).astype(np.float32)
    sum_x = np.bincount(
        label_values,
        weights=x_grid[foreground].reshape(-1),
        minlength=int(label_count),
    ).astype(np.float32)
    sum_y = np.bincount(
        label_values,
        weights=y_grid[foreground].reshape(-1),
        minlength=int(label_count),
    ).astype(np.float32)
    centroid_x = np.divide(sum_x, areas, out=np.zeros_like(sum_x), where=areas > 0)
    centroid_y = np.divide(sum_y, areas, out=np.zeros_like(sum_y), where=areas > 0)
    valid = np.arange(int(label_count), dtype=np.int32) > 0
    valid &= areas > 0
    distance = (centroid_x - float(point[0])) ** 2 + (centroid_y - float(point[1])) ** 2
    distance[~valid] = np.inf
    best_label = int(np.argmin(distance))

    if best_label > 0 and bool(np.isfinite(distance[best_label])):
        selected = labels == best_label
    return selected


def coordinate_grids(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return cached x/y float32 coordinate grids for a 2D mask shape."""

    if len(shape) != 2:
        raise ValueError(f"Expected 2D mask shape, got {shape!r}.")
    key = (int(shape[0]), int(shape[1]))
    cached = _COORD_GRID_CACHE.get(key)
    if cached is None:
        y_grid, x_grid = np.indices(key, dtype=np.float32)
        cached = (x_grid, y_grid)
        _COORD_GRID_CACHE[key] = cached
    return cached


def mask_pixel_centroid(mask: np.ndarray) -> np.ndarray:
    """Return the foreground pixel centroid as ``[x, y]`` or NaNs if empty."""

    ys, xs = np.nonzero(mask.astype(np.uint8))
    if ys.size > 0:
        return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
    return np.full(2, np.nan, dtype=np.float32)
