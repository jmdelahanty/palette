from __future__ import annotations

import errno

import pytest

from fisheye.shared.transient_io import retry_read_only_estale


def test_retry_read_only_estale_uses_bounded_delay_schedule() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError(errno.ESTALE, "stale file handle")
        return "ok"

    assert retry_read_only_estale(
        operation,
        delays_seconds=(0.1, 0.25, 0.5),
        sleep=sleeps.append,
    ) == "ok"
    assert calls == 3
    assert sleeps == [0.1, 0.25]


def test_retry_read_only_estale_does_not_retry_other_os_errors() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise OSError(errno.EIO, "I/O error")

    with pytest.raises(OSError) as exc_info:
        retry_read_only_estale(
            operation,
            delays_seconds=(0.1, 0.25),
            sleep=sleeps.append,
        )

    assert exc_info.value.errno == errno.EIO
    assert calls == 1
    assert sleeps == []


def test_retry_read_only_estale_surfaces_error_after_budget_exhaustion() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise OSError(errno.ESTALE, "stale file handle")

    with pytest.raises(OSError) as exc_info:
        retry_read_only_estale(
            operation,
            delays_seconds=(0.1, 0.25),
            sleep=sleeps.append,
        )

    assert exc_info.value.errno == errno.ESTALE
    assert calls == 3
    assert sleeps == [0.1, 0.25]
