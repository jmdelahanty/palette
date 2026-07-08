from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.plot_sampled_component_contours import (
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
