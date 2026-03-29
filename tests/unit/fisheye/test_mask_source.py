from __future__ import annotations

import numpy as np

from fisheye.shared.mask_source import _to_float01


def test_to_float01_dequantizes_uint8_probabilities() -> None:
    arr = np.array([0, 128, 255], dtype=np.uint8)

    out = _to_float01(arr)

    assert out.dtype == np.float32
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[1], 128.0 / 255.0)
    assert np.isclose(out[2], 1.0)
