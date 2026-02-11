"""Helpers for enforcing row-level alignment across stage inputs."""

from __future__ import annotations

from typing import Iterable, Tuple


def assert_row_alignment(
    expected_rows: int,
    named_arrays: Iterable[Tuple[str, object]],
    *,
    stage: str,
) -> None:
    """Raise when any provided array's leading dimension differs from expected_rows."""
    mismatches = []
    for name, array in named_arrays:
        if array is None:
            continue
        shape = getattr(array, "shape", None)
        if shape is None:
            mismatches.append(f"{name}=<missing-shape>")
            continue
        if len(shape) == 0:
            mismatches.append(f"{name}=<scalar>")
            continue
        actual = int(shape[0])
        if actual != expected_rows:
            mismatches.append(f"{name}={actual}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            f"{stage} row-alignment check failed (expected {expected_rows} rows): {joined}"
        )
