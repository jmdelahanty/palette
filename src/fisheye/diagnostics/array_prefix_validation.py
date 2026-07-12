"""Semantic helpers for validating array prefixes across differently sized runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ZeroPaddedBytePrefixComparison:
    """Result of comparing fixed-width byte rows modulo trailing zero padding."""

    equal: bool
    reason: str | None
    row_count: int
    reference_width: int | None
    candidate_width: int | None
    common_width: int | None
    reference_extra_zero: bool | None
    candidate_extra_zero: bool | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def compare_zero_padded_uint8_row_prefix(
    reference: Any,
    candidate: Any,
    *,
    row_count: int,
    row_step: int,
) -> ZeroPaddedBytePrefixComparison:
    """Compare ``uint8[N,width]`` rows while ignoring only trailing zero columns.

    The candidate may contain additional rows because it represents a full
    collection while ``reference`` represents a clip prefix. Width differences
    are semantic padding only when shared columns match exactly and every extra
    byte on either side is zero.
    """

    row_count = int(row_count)
    row_step = int(row_step)
    if row_count < 0:
        raise ValueError("row_count must be non-negative.")
    if row_step <= 0:
        raise ValueError("row_step must be positive.")

    reference_shape = tuple(int(value) for value in reference.shape)
    candidate_shape = tuple(int(value) for value in candidate.shape)
    reference_dtype = np.dtype(reference.dtype)
    candidate_dtype = np.dtype(candidate.dtype)
    if reference_dtype != np.dtype(np.uint8) or candidate_dtype != np.dtype(np.uint8):
        return ZeroPaddedBytePrefixComparison(
            equal=False,
            reason="dtype must be uint8",
            row_count=row_count,
            reference_width=None,
            candidate_width=None,
            common_width=None,
            reference_extra_zero=None,
            candidate_extra_zero=None,
        )
    if len(reference_shape) != 2 or len(candidate_shape) != 2:
        return ZeroPaddedBytePrefixComparison(
            equal=False,
            reason="arrays must be two-dimensional",
            row_count=row_count,
            reference_width=None,
            candidate_width=None,
            common_width=None,
            reference_extra_zero=None,
            candidate_extra_zero=None,
        )

    reference_width = int(reference_shape[1])
    candidate_width = int(candidate_shape[1])
    common_width = min(reference_width, candidate_width)
    if int(reference_shape[0]) != row_count:
        return ZeroPaddedBytePrefixComparison(
            equal=False,
            reason=f"reference row count {reference_shape[0]} != expected prefix {row_count}",
            row_count=row_count,
            reference_width=reference_width,
            candidate_width=candidate_width,
            common_width=common_width,
            reference_extra_zero=None,
            candidate_extra_zero=None,
        )
    if int(candidate_shape[0]) < row_count:
        return ZeroPaddedBytePrefixComparison(
            equal=False,
            reason=f"candidate row count {candidate_shape[0]} < expected prefix {row_count}",
            row_count=row_count,
            reference_width=reference_width,
            candidate_width=candidate_width,
            common_width=common_width,
            reference_extra_zero=None,
            candidate_extra_zero=None,
        )

    reference_extra_zero = True
    candidate_extra_zero = True
    for start in range(0, row_count, row_step):
        stop = min(row_count, start + row_step)
        reference_rows = np.asarray(reference[start:stop], dtype=np.uint8)
        candidate_rows = np.asarray(candidate[start:stop], dtype=np.uint8)
        if not np.array_equal(
            reference_rows[:, :common_width],
            candidate_rows[:, :common_width],
        ):
            return ZeroPaddedBytePrefixComparison(
                equal=False,
                reason=f"shared bytes differ in prefix rows {start}:{stop}",
                row_count=row_count,
                reference_width=reference_width,
                candidate_width=candidate_width,
                common_width=common_width,
                reference_extra_zero=reference_extra_zero,
                candidate_extra_zero=candidate_extra_zero,
            )
        if reference_width > common_width and bool(np.any(reference_rows[:, common_width:])):
            reference_extra_zero = False
            return ZeroPaddedBytePrefixComparison(
                equal=False,
                reason=f"reference trailing bytes are nonzero in prefix rows {start}:{stop}",
                row_count=row_count,
                reference_width=reference_width,
                candidate_width=candidate_width,
                common_width=common_width,
                reference_extra_zero=False,
                candidate_extra_zero=candidate_extra_zero,
            )
        if candidate_width > common_width and bool(np.any(candidate_rows[:, common_width:])):
            candidate_extra_zero = False
            return ZeroPaddedBytePrefixComparison(
                equal=False,
                reason=f"candidate trailing bytes are nonzero in prefix rows {start}:{stop}",
                row_count=row_count,
                reference_width=reference_width,
                candidate_width=candidate_width,
                common_width=common_width,
                reference_extra_zero=reference_extra_zero,
                candidate_extra_zero=False,
            )

    return ZeroPaddedBytePrefixComparison(
        equal=True,
        reason=None,
        row_count=row_count,
        reference_width=reference_width,
        candidate_width=candidate_width,
        common_width=common_width,
        reference_extra_zero=reference_extra_zero,
        candidate_extra_zero=candidate_extra_zero,
    )


__all__ = [
    "ZeroPaddedBytePrefixComparison",
    "compare_zero_padded_uint8_row_prefix",
]
