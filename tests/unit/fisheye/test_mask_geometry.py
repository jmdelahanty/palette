from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.ndimage import binary_fill_holes

from fisheye.shared.mask_geometry import (
    fill_holes,
    hole_stats,
    mask_pixel_centroid,
    measure_mask_ellipse,
    select_component_near_point,
)


@pytest.mark.parametrize("case", ["empty", "solid", "hole", "border_touching", "two_holes"])
def test_fill_holes_matches_scipy_binary_fill_holes(case: str) -> None:
    mask = np.zeros((32, 32), dtype=bool)
    if case == "solid":
        mask[8:24, 8:24] = True
    elif case == "hole":
        mask[8:24, 8:24] = True
        mask[14:18, 14:18] = False
    elif case == "border_touching":
        mask[0:24, 0:24] = True
        mask[8:14, 8:14] = False
    elif case == "two_holes":
        mask[4:28, 4:28] = True
        mask[8:12, 8:12] = False
        mask[18:23, 18:25] = False

    expected = np.asarray(binary_fill_holes(mask), dtype=bool)
    actual = fill_holes(mask)

    assert np.array_equal(actual, expected)


def test_hole_stats_reports_count_fraction_and_area() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 2:18] = True
    mask[5:8, 5:9] = False
    mask[12:15, 12:16] = False

    count, fraction, area = hole_stats(mask)

    assert count == 2
    assert area == 24
    assert fraction == pytest.approx(24 / float(np.count_nonzero(mask) + 24))


def test_select_component_near_point_returns_single_component_mask() -> None:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[8:14, 7:16] = 1

    selected = select_component_near_point(mask, np.asarray([12.0, 10.0], dtype=np.float32))

    np.testing.assert_array_equal(selected, mask.astype(bool))


def test_select_component_near_point_matches_legacy_stats_selector() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:9, 4:10] = 1
    mask[12:19, 18:27] = 1
    mask[22:27, 5:12] = 1
    point = np.asarray([22.0, 15.0], dtype=np.float32)

    selected = select_component_near_point(mask, point)
    expected = _legacy_select_component_near_point(mask, point)

    np.testing.assert_array_equal(selected, expected)


def test_mask_pixel_centroid_returns_xy_or_nan() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2, 3] = 1
    mask[4, 7] = 1

    np.testing.assert_allclose(mask_pixel_centroid(mask), np.asarray([5.0, 3.0], dtype=np.float32))
    assert np.all(np.isnan(mask_pixel_centroid(np.zeros((4, 4), dtype=np.uint8))))


def test_measure_mask_ellipse_rejects_subpixel_degenerate_fit() -> None:
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[275, 248] = 1
    mask[276:281, 247] = 1
    mask[279, 246] = 1

    success, ellipse, centroid, contour, failure = measure_mask_ellipse(mask)

    assert success is False
    assert failure == "ellipse_invalid_params"
    assert np.all(np.isnan(ellipse))
    np.testing.assert_allclose(centroid, mask_pixel_centroid(mask))
    assert contour is not None
    assert contour.shape == (10, 2)


def test_measure_mask_ellipse_accepts_resolved_pixel_scale_ellipse() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.ellipse(mask, (32, 31), (8, 4), 20, 0, 360, 1, -1)

    success, ellipse, centroid, contour, failure = measure_mask_ellipse(mask)

    assert success is True
    assert failure is None
    assert contour is not None
    assert np.all(np.isfinite(ellipse))
    assert ellipse[2] >= ellipse[3] >= 1.0
    assert np.all(np.isfinite(centroid))


def _legacy_select_component_near_point(mask: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    selected = np.zeros_like(binary, dtype=bool)
    if int(np.count_nonzero(binary)) <= 0:
        return selected
    label_count, labels, _stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if label_count <= 1:
        return selected
    point = np.asarray(point_xy, dtype=np.float32).reshape(-1)
    best_label = 0
    best_distance = float("inf")
    for label_idx in range(1, int(label_count)):
        centroid = np.asarray(centroids[label_idx], dtype=np.float32)
        distance = float(np.sum(np.square(centroid[:2] - point[:2], dtype=np.float32), dtype=np.float32))
        if distance < best_distance:
            best_distance = distance
            best_label = int(label_idx)
    if best_label > 0:
        selected = labels == best_label
    return selected
