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


def batch_mask_spatial_metrics(masks: np.ndarray) -> dict[str, np.ndarray]:
    """Compute area, centroid, and bbox metrics for a stack of binary masks.

    The final two axes are interpreted as ``(H, W)``. Any leading dimensions are
    preserved, so callers can pass either ``(N,H,W)`` component masks or
    ``(N,C,H,W)`` multi-component masks. Bounding boxes use half-open pixel-edge
    bounds ``[x_min, y_min, x_max_exclusive, y_max_exclusive]``.
    """

    binary = np.asarray(masks, dtype=np.uint8) > 0
    if binary.ndim < 3:
        raise ValueError(f"Expected masks with at least 3 dimensions (...,H,W), got {tuple(binary.shape)}.")
    leading_shape = tuple(int(dim) for dim in binary.shape[:-2])
    height = int(binary.shape[-2])
    width = int(binary.shape[-1])
    flat_count = int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1
    flat = binary.reshape(flat_count, height, width)

    area_px_flat = flat.reshape(flat_count, -1).sum(axis=1, dtype=np.int64).astype(np.float32)
    mask_present_flat = area_px_flat > 0.0
    centroid_xy_flat = np.zeros((flat_count, 2), dtype=np.float32)
    bbox_xyxy_flat = np.zeros((flat_count, 4), dtype=np.float32)

    if flat_count > 0 and bool(np.any(mask_present_flat)):
        y_counts = flat.sum(axis=2, dtype=np.float32)
        x_counts = flat.sum(axis=1, dtype=np.float32)
        y_coords = np.arange(height, dtype=np.float32)
        x_coords = np.arange(width, dtype=np.float32)
        denominator = np.maximum(area_px_flat, 1.0).astype(np.float32, copy=False)
        centroid_xy_flat[:, 0] = np.asarray(x_counts @ x_coords, dtype=np.float32) / denominator
        centroid_xy_flat[:, 1] = np.asarray(y_counts @ y_coords, dtype=np.float32) / denominator
        centroid_xy_flat[~mask_present_flat] = 0.0

        row_has_mask = flat.any(axis=2)
        col_has_mask = flat.any(axis=1)
        y_indices = np.arange(height, dtype=np.int32).reshape(1, height)
        x_indices = np.arange(width, dtype=np.int32).reshape(1, width)
        y_min = np.where(row_has_mask, y_indices, height).min(axis=1)
        y_max_exclusive = np.where(row_has_mask, y_indices + 1, 0).max(axis=1)
        x_min = np.where(col_has_mask, x_indices, width).min(axis=1)
        x_max_exclusive = np.where(col_has_mask, x_indices + 1, 0).max(axis=1)
        bbox_xyxy_flat[:, 0] = x_min.astype(np.float32, copy=False)
        bbox_xyxy_flat[:, 1] = y_min.astype(np.float32, copy=False)
        bbox_xyxy_flat[:, 2] = x_max_exclusive.astype(np.float32, copy=False)
        bbox_xyxy_flat[:, 3] = y_max_exclusive.astype(np.float32, copy=False)
        bbox_xyxy_flat[~mask_present_flat] = 0.0

    return {
        "mask_present": mask_present_flat.reshape(leading_shape).astype(bool, copy=False),
        "area_px": area_px_flat.reshape(leading_shape).astype(np.float32, copy=False),
        "centroid_xy": centroid_xy_flat.reshape((*leading_shape, 2)).astype(np.float32, copy=False),
        "centroid_valid": mask_present_flat.reshape(leading_shape).astype(bool, copy=False),
        "bbox_xyxy": bbox_xyxy_flat.reshape((*leading_shape, 4)).astype(np.float32, copy=False),
        "bbox_valid": mask_present_flat.reshape(leading_shape).astype(bool, copy=False),
    }


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


def extract_mask_contour(mask: np.ndarray, min_points: int) -> np.ndarray | None:
    """Return the largest external contour for a binary mask in ``(x, y)`` order."""

    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if contour.shape[0] < min_points:
        return None
    return contour


def measure_mask_ellipse(
    mask: np.ndarray,
    min_contour_points: int = 5,
    min_axis_length_px: float = 1.0,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray | None, str | None]:
    """Extract ellipse metrics from a binary mask using OpenCV ellipse fitting."""

    if mask.sum() == 0:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        centroid = np.full(2, np.nan, dtype=np.float32)
        return False, ellipse, centroid, None, "empty_mask"

    contour = extract_mask_contour(mask.astype(float), min_contour_points)
    if contour is None:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        centroid = np.full(2, np.nan, dtype=np.float32)
        return False, ellipse, centroid, None, "contour_missing"

    contour = contour.astype(np.float32)

    try:
        (xc, yc), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    except cv2.error:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        centroid = mask_pixel_centroid(mask)
        return False, ellipse, centroid, contour, "ellipse_estimate_failed"

    major = float(axis_a)
    minor = float(axis_b)
    theta = float(angle)
    if major < minor:
        major, minor = minor, major
        theta += 90.0
    theta = float((theta + 180.0) % 180.0)

    minimum_axis = float(min_axis_length_px)
    if not np.isfinite(minimum_axis) or minimum_axis < 0.0:
        raise ValueError(f"min_axis_length_px must be finite and nonnegative, got {min_axis_length_px!r}.")
    if (
        not all(np.isfinite([xc, yc, major, minor, theta]))
        or major <= 0.0
        or minor <= 0.0
        or major < minimum_axis
        or minor < minimum_axis
    ):
        ellipse = np.full(5, np.nan, dtype=np.float32)
        centroid = mask_pixel_centroid(mask)
        return False, ellipse, centroid, contour, "ellipse_invalid_params"

    centroid = np.array([float(xc), float(yc)], dtype=np.float32)
    ellipse = np.array(
        [
            float(xc),
            float(yc),
            major,
            minor,
            theta,
        ],
        dtype=np.float32,
    )

    return True, ellipse, centroid, contour, None
