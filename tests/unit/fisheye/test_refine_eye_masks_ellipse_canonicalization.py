from __future__ import annotations

import math

import numpy as np
import pytest

from fisheye.refinement import refine_eye_masks as mod


def test_measure_mask_canonicalizes_major_minor_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    contour = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 1.5],
        ],
        dtype=np.float32,
    )

    class _FakeEllipseModel:
        def __init__(self) -> None:
            self.params: tuple[float, float, float, float, float] | None = None

        def estimate(self, _points: np.ndarray) -> bool:
            # Deliberately return a < b to verify canonicalization.
            self.params = (4.0, 5.0, 3.0, 7.0, 0.25)
            return True

    monkeypatch.setattr(mod, "_extract_contour", lambda _mask, _min_points=5: contour)
    monkeypatch.setattr(mod, "EllipseModel", _FakeEllipseModel)

    success, ellipse, centroid, contour_out = mod._measure_mask(np.ones((8, 8), dtype=np.uint8))

    assert success is True
    assert contour_out is not None
    np.testing.assert_allclose(centroid, np.array([4.0, 5.0], dtype=np.float32))
    # Full axis lengths should be major=14, minor=6 after canonicalization.
    assert ellipse[2] == pytest.approx(14.0)
    assert ellipse[3] == pytest.approx(6.0)
    assert ellipse[4] == pytest.approx(np.rad2deg(0.25 + (math.pi / 2.0)))

