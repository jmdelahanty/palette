from __future__ import annotations

import numpy as np
import pytest

from fisheye.refinement import refine_eye_masks as mod


def test_measure_mask_canonicalizes_major_minor_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 1.5]], dtype=np.float32)

    monkeypatch.setattr(mod, "_extract_contour", lambda _mask, _min_points=5: contour)
    monkeypatch.setattr(mod.cv2, "fitEllipse", lambda _points: ((4.0, 5.0), (3.0, 7.0), 20.0))

    success, ellipse, centroid, contour_out, failure_reason = mod._measure_mask(
        np.ones((8, 8), dtype=np.uint8)
    )

    assert success is True
    assert failure_reason is None
    assert contour_out is not None
    np.testing.assert_allclose(centroid, np.array([4.0, 5.0], dtype=np.float32))
    assert ellipse[2] == pytest.approx(7.0)
    assert ellipse[3] == pytest.approx(3.0)
    assert ellipse[4] == pytest.approx(110.0)


def test_measure_mask_reports_fit_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 1.5]], dtype=np.float32)
    monkeypatch.setattr(mod, "_extract_contour", lambda _mask, _min_points=5: contour)

    def _raise_fit(_points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
        raise mod.cv2.error("fitEllipse", "", "", -1)

    monkeypatch.setattr(mod.cv2, "fitEllipse", _raise_fit)

    success, ellipse, centroid, contour_out, failure_reason = mod._measure_mask(
        np.ones((8, 8), dtype=np.uint8)
    )
    assert success is False
    assert failure_reason == "ellipse_estimate_failed"
    assert contour_out is not None
    assert np.all(np.isfinite(centroid))
    assert np.all(np.isnan(ellipse))


def test_measure_mask_reports_invalid_params(monkeypatch: pytest.MonkeyPatch) -> None:
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 1.5]], dtype=np.float32)
    monkeypatch.setattr(mod, "_extract_contour", lambda _mask, _min_points=5: contour)
    monkeypatch.setattr(mod.cv2, "fitEllipse", lambda _points: ((4.0, 5.0), (0.0, 7.0), 20.0))

    success, ellipse, centroid, contour_out, failure_reason = mod._measure_mask(
        np.ones((8, 8), dtype=np.uint8)
    )

    assert success is False
    assert failure_reason == "ellipse_invalid_params"
    assert contour_out is not None
    assert np.all(np.isfinite(centroid))
    assert np.all(np.isnan(ellipse))


def test_append_ellipse_failure_tags_marks_left_right_and_pair() -> None:
    tagged = mod._append_ellipse_failure_tags(
        "union_source|split_by_keypoint",
        left_failure="ellipse_invalid_params",
        right_failure="contour_missing",
    )

    assert tagged is not None
    parts = set(tagged.split("|"))
    assert "union_source" in parts
    assert "split_by_keypoint" in parts
    assert "ellipse_fail_left" in parts
    assert "ellipse_fail_right" in parts
    assert "ellipse_fail_pair" in parts
    assert "ellipse_invalid_params_left" in parts
    assert "contour_missing_right" in parts
