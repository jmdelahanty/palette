from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_PROFILE_ID
from fisheye.shared.subject_position_types import (
    CANONICAL_FLOAT32_QNAN_BITS,
    POSITION_FAILURE_REASON_CODES,
    POSITION_FAILURE_REASON_PRECEDENCE,
    POSITION_FAILURE_REASON_TAGS,
    SOURCE_CAMERA_POSITION_PROFILE_ID,
    canonical_float32_nan,
    empty_position_xy,
)


def test_position_failure_reason_registry_is_bijective() -> None:
    assert POSITION_FAILURE_REASON_CODES["ok"] == 0
    assert len(POSITION_FAILURE_REASON_CODES) == len(POSITION_FAILURE_REASON_TAGS)
    assert {
        code: tag for tag, code in POSITION_FAILURE_REASON_CODES.items()
    } == dict(POSITION_FAILURE_REASON_TAGS)
    assert set(POSITION_FAILURE_REASON_PRECEDENCE) == (
        set(POSITION_FAILURE_REASON_CODES) - {"ok"}
    )
    assert len(POSITION_FAILURE_REASON_PRECEDENCE) == len(
        set(POSITION_FAILURE_REASON_PRECEDENCE)
    )


def test_position_profile_reuses_canonical_coordinate_authority() -> None:
    assert SOURCE_CAMERA_POSITION_PROFILE_ID is SOURCE_CAMERA_PROFILE_ID


def test_canonical_float32_nan_uses_exact_contract_bits() -> None:
    value = np.asarray(canonical_float32_nan(), dtype=np.float32).reshape(())
    assert np.isnan(value)
    assert value.view(np.uint32)[()] == CANONICAL_FLOAT32_QNAN_BITS


@pytest.mark.parametrize("row_count", [0, 1, 5])
def test_empty_position_xy_uses_exact_shape_dtype_and_nan_bits(row_count: int) -> None:
    values = empty_position_xy(row_count)
    assert values.shape == (row_count, 2)
    assert values.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        values.view(np.uint32),
        np.full((row_count, 2), CANONICAL_FLOAT32_QNAN_BITS, dtype=np.uint32),
    )


def test_empty_position_xy_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        empty_position_xy(-1)
