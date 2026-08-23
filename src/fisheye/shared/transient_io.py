"""Bounded recovery helpers for transient read-only filesystem faults."""

from __future__ import annotations

import errno
import time
from collections.abc import Callable, Sequence
from typing import TypeVar


T = TypeVar("T")

DEFAULT_ESTALE_RETRY_DELAYS_SECONDS: tuple[float, ...] = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)


def retry_read_only_estale(
    operation: Callable[[], T],
    *,
    delays_seconds: Sequence[float] = DEFAULT_ESTALE_RETRY_DELAYS_SECONDS,
    sleep: Callable[[float], object] = time.sleep,
) -> T:
    """Retry an idempotent read when NFS reports ``ESTALE``.

    Callers must pass a read-only operation. Other errors are surfaced without
    retry, and the original ``ESTALE`` error is surfaced after the bounded
    delay schedule is exhausted.
    """

    delays = tuple(float(value) for value in delays_seconds)
    if any(value < 0 for value in delays):
        raise ValueError("ESTALE retry delays cannot be negative.")
    for retry_index in range(len(delays) + 1):
        try:
            return operation()
        except OSError as exc:
            if exc.errno != errno.ESTALE or retry_index == len(delays):
                raise
            sleep(delays[retry_index])
    raise AssertionError("unreachable ESTALE retry state")


__all__ = [
    "DEFAULT_ESTALE_RETRY_DELAYS_SECONDS",
    "retry_read_only_estale",
]
