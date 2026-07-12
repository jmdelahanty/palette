from __future__ import annotations

import numpy as np

from fisheye.diagnostics.array_prefix_validation import compare_zero_padded_uint8_row_prefix


def compare(reference: np.ndarray, candidate: np.ndarray, *, rows: int | None = None):
    return compare_zero_padded_uint8_row_prefix(
        reference,
        candidate,
        row_count=int(reference.shape[0] if rows is None else rows),
        row_step=2,
    )


def test_wider_candidate_zero_padding_is_equal() -> None:
    reference = np.asarray([[1, 2, 0], [3, 0, 0], [4, 5, 6]], dtype=np.uint8)
    candidate = np.zeros((5, 6), dtype=np.uint8)
    candidate[:3, :3] = reference

    result = compare(reference, candidate)

    assert result.equal
    assert result.reference_width == 3
    assert result.candidate_width == 6
    assert result.common_width == 3
    assert result.reference_extra_zero is True
    assert result.candidate_extra_zero is True


def test_narrower_candidate_is_equal_when_reference_extra_is_zero() -> None:
    reference = np.asarray([[1, 2, 0, 0], [3, 0, 0, 0]], dtype=np.uint8)
    candidate = np.asarray([[1, 2], [3, 0], [9, 9]], dtype=np.uint8)

    result = compare(reference, candidate)

    assert result.equal
    assert result.reference_width == 4
    assert result.candidate_width == 2
    assert result.reference_extra_zero is True


def test_nonzero_candidate_padding_fails() -> None:
    reference = np.asarray([[1, 2], [3, 0]], dtype=np.uint8)
    candidate = np.asarray([[1, 2, 0], [3, 0, 7]], dtype=np.uint8)

    result = compare(reference, candidate)

    assert not result.equal
    assert result.candidate_extra_zero is False
    assert result.reason == "candidate trailing bytes are nonzero in prefix rows 0:2"


def test_nonzero_reference_padding_fails() -> None:
    reference = np.asarray([[1, 2, 0], [3, 0, 7]], dtype=np.uint8)
    candidate = np.asarray([[1, 2], [3, 0], [9, 9]], dtype=np.uint8)

    result = compare(reference, candidate)

    assert not result.equal
    assert result.reference_extra_zero is False
    assert result.reason == "reference trailing bytes are nonzero in prefix rows 0:2"


def test_shared_byte_mismatch_fails() -> None:
    reference = np.asarray([[1, 2], [3, 0]], dtype=np.uint8)
    candidate = np.asarray([[1, 2, 0], [4, 0, 0]], dtype=np.uint8)

    result = compare(reference, candidate)

    assert not result.equal
    assert result.reason == "shared bytes differ in prefix rows 0:2"


def test_candidate_must_contain_full_prefix() -> None:
    reference = np.zeros((3, 2), dtype=np.uint8)
    candidate = np.zeros((2, 2), dtype=np.uint8)

    result = compare(reference, candidate)

    assert not result.equal
    assert result.reason == "candidate row count 2 < expected prefix 3"


def test_non_uint8_arrays_fail_closed() -> None:
    reference = np.zeros((2, 2), dtype=np.int16)
    candidate = np.zeros((2, 3), dtype=np.int16)

    result = compare(reference, candidate)

    assert not result.equal
    assert result.reason == "dtype must be uint8"
