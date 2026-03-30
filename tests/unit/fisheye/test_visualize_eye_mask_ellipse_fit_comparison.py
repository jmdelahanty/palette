from __future__ import annotations

import numpy as np

from fisheye.visualization import visualize_eye_mask_ellipse_fit_comparison as mod


def test_compute_fit_comparison_reports_empty_mask_failure() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)

    cv2_fit, sk_fit = mod._compute_fit_comparison(mask)

    assert not cv2_fit.success
    assert not sk_fit.success
    assert sk_fit.failure_reason == "empty_mask"


def test_compute_fit_comparison_returns_param_arrays_for_valid_mask() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    yy, xx = np.ogrid[:32, :32]
    ellipse = ((xx - 16.0) ** 2) / (8.0**2) + ((yy - 16.0) ** 2) / (5.0**2) <= 1.0
    mask[ellipse] = 1

    cv2_fit, sk_fit = mod._compute_fit_comparison(mask)

    assert cv2_fit.params.shape == (5,)
    assert sk_fit.params.shape == (5,)


def test_should_draw_fit_uses_component_visibility_map() -> None:
    visibility = {
        "cv2_contour": True,
        "cv2_ellipse": False,
        "skimage_contour": False,
        "skimage_ellipse": True,
    }

    assert mod._should_draw_fit("cv2", "contour", visibility)
    assert not mod._should_draw_fit("cv2", "ellipse", visibility)
    assert not mod._should_draw_fit("skimage", "contour", visibility)
    assert mod._should_draw_fit("skimage", "ellipse", visibility)
    assert not mod._should_draw_fit("unknown", "ellipse", visibility)
