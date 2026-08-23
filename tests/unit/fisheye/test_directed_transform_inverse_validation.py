from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.directed_transform_v2 import (
    DirectedTransformV2Error,
    _validate_inverse_pair,
)


FORWARD = np.asarray(
    [
        [0.0790686027, -0.000477930292, 542.998478],
        [-0.000547065407, -0.0799886753, 518.463643],
        [-0.000000459955865, -0.000000388605225, 1.0],
    ],
    dtype=np.float64,
)


def test_inverse_validation_accepts_one_ulp_persisted_difference() -> None:
    inverse = np.linalg.inv(FORWARD)
    inverse[0, 0] = np.nextafter(inverse[0, 0], np.inf)

    _validate_inverse_pair(FORWARD, inverse)


def test_inverse_validation_does_not_recompute_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inverse = np.linalg.inv(FORWARD)
    monkeypatch.setattr(
        np.linalg,
        "inv",
        lambda _matrix: (_ for _ in ()).throw(AssertionError("reinversion forbidden")),
    )

    _validate_inverse_pair(FORWARD, inverse)


def test_inverse_validation_rejects_meaningful_mismatch() -> None:
    inverse = np.linalg.inv(FORWARD)
    inverse[0, 2] += 1e-4

    with pytest.raises(
        DirectedTransformV2Error,
        match="do not compose to identity",
    ):
        _validate_inverse_pair(FORWARD, inverse)
