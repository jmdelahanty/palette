from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.mask_probability_encoding import decode_probability_values_from_attrs


def test_decode_probability_values_from_attrs_decodes_uint8_linear() -> None:
    decoded = decode_probability_values_from_attrs(
        np.asarray([0, 128, 255], dtype=np.uint8),
        attrs={"probabilities_encoding": "linear_uint8_0_255"},
        source_path="subject_mask_runs/run/mask_probs_roi",
    )

    np.testing.assert_allclose(decoded, np.asarray([0.0, 128.0 / 255.0, 1.0], dtype=np.float32))


def test_decode_probability_values_from_attrs_requires_encoding() -> None:
    with pytest.raises(ValueError, match="subject_mask_runs/run/mask_probs_roi.*missing.*uint8"):
        decode_probability_values_from_attrs(
            np.asarray([0, 255], dtype=np.uint8),
            attrs={},
            source_path="subject_mask_runs/run/mask_probs_roi",
        )


def test_decode_probability_values_from_attrs_rejects_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="unrecognized value 'legacy_guess'.*float16"):
        decode_probability_values_from_attrs(
            np.asarray([0.0, 1.0], dtype=np.float16),
            attrs={"probabilities_encoding": "legacy_guess"},
            source_path="subject_mask_runs/run/mask_probs_roi",
        )


def test_decode_probability_values_from_attrs_rejects_dtype_encoding_mismatch() -> None:
    with pytest.raises(ValueError, match="requires uint8 storage.*float16"):
        decode_probability_values_from_attrs(
            np.asarray([0.0, 1.0], dtype=np.float16),
            attrs={"probabilities_encoding": "linear_uint8_0_255"},
            source_path="subject_mask_runs/run/mask_probs_roi",
        )
