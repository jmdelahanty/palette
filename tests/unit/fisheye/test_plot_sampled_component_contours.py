from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.plot_sampled_component_contours import (
    RoiImageSource,
    _normalize_image,
    component_k,
    parse_component_k,
    resample_closed_polyline,
)


def test_resample_closed_polyline_returns_fixed_k_samples() -> None:
    square = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )

    sampled = resample_closed_polyline(square, 8)

    assert sampled.shape == (8, 2)
    np.testing.assert_allclose(sampled[0], [0.0, 0.0])
    np.testing.assert_allclose(sampled[2], [10.0, 0.0])
    np.testing.assert_allclose(sampled[4], [10.0, 10.0])
    np.testing.assert_allclose(sampled[6], [0.0, 10.0])


def test_resample_closed_polyline_handles_degenerate_contours() -> None:
    sampled_empty = resample_closed_polyline(np.empty((0, 2), dtype=np.float32), 3)
    sampled_single = resample_closed_polyline(np.asarray([[4.0, 5.0]], dtype=np.float32), 3)

    assert sampled_empty.shape == (3, 2)
    assert np.isnan(sampled_empty).all()
    np.testing.assert_allclose(sampled_single, np.asarray([[4.0, 5.0]] * 3, dtype=np.float32))


def test_parse_component_k_and_defaults() -> None:
    overrides = parse_component_k(["subject_body=512", "eye_left=48"])

    assert component_k("subject_body", overrides, default_k=32) == 512
    assert component_k("eye_left", overrides, default_k=32) == 48
    assert component_k("eye_right", overrides, default_k=32) == 64
    assert component_k("unknown_component", overrides, default_k=32) == 32


def test_parse_component_k_rejects_invalid_specs() -> None:
    with pytest.raises(Exception, match="Expected COMPONENT=K"):
        parse_component_k(["subject_body"])
    with pytest.raises(Exception, match="K must be positive"):
        parse_component_k(["subject_body=0"])


def test_roi_image_source_maps_refined_rows_to_crop_rows() -> None:
    roi_images = np.stack(
        [
            np.full((4, 4), 10, dtype=np.uint8),
            np.full((4, 4), 90, dtype=np.uint8),
        ],
        axis=0,
    )
    source = RoiImageSource(
        crop_run_name="crop_test",
        roi_images=roi_images,
        source_crop_row_ids=np.asarray([1, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(source.image_for_refined_row(0), roi_images[1])
    np.testing.assert_array_equal(source.image_for_refined_row(1), roi_images[0])
    assert source.image_for_refined_row(-1) is None
    assert source.image_for_refined_row(2) is None


def test_normalize_image_uses_robust_display_range() -> None:
    image = np.asarray([[0, 10, 20], [30, 40, 255]], dtype=np.uint8)

    normalized = _normalize_image(image)

    assert normalized.shape == image.shape
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0
