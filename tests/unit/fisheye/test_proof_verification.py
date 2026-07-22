from __future__ import annotations

import pytest

from fisheye.shared.proof_verification import (
    finish_proof_verification,
    load_verified_value,
    proof_verification_operation,
    proof_verification_scope,
    restart_proof_verification,
    verify_persisted_proof,
)


def test_outside_scope_every_proof_is_fresh() -> None:
    calls: list[str] = []

    verify_persisted_proof(("authority", "digest"), lambda: calls.append("verify"))
    verify_persisted_proof(("authority", "digest"), lambda: calls.append("verify"))

    assert calls == ["verify", "verify"]


def test_scope_reuses_exact_key_and_rechecks_before_return() -> None:
    calls: list[str] = []

    def verify() -> None:
        calls.append("verify")

    with proof_verification_scope():
        verify_persisted_proof(("authority", "digest"), verify)
        verify_persisted_proof(("authority", "digest"), verify)
        verify_persisted_proof(("authority", "digest"), verify)
        assert calls == ["verify"]

    assert calls == ["verify", "verify"]


def test_nested_operations_share_only_the_outer_scope() -> None:
    calls: list[str] = []

    @proof_verification_operation
    def inner() -> None:
        verify_persisted_proof(("authority", "digest"), lambda: calls.append("verify"))

    @proof_verification_operation
    def outer() -> None:
        inner()
        inner()
        assert calls == ["verify"]

    outer()
    assert calls == ["verify", "verify"]

    inner()
    assert calls == ["verify", "verify", "verify", "verify"]


def test_distinct_digest_proofs_are_never_coalesced() -> None:
    calls: list[str] = []

    with proof_verification_scope():
        verify_persisted_proof(("authority", "digest-a"), lambda: calls.append("a"))
        verify_persisted_proof(("authority", "digest-b"), lambda: calls.append("b"))

    assert calls == ["a", "b", "a", "b"]


def test_failed_initial_proof_is_not_cached() -> None:
    attempts = 0

    def verify() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("stale")

    with proof_verification_scope():
        with pytest.raises(ValueError, match="stale"):
            verify_persisted_proof(("authority", "digest"), verify)
        verify_persisted_proof(("authority", "digest"), verify)

    assert attempts == 3


def test_closing_recheck_failure_fails_the_operation_and_clears_scope() -> None:
    attempts = 0

    def verify() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise ValueError("changed during operation")

    with pytest.raises(ValueError, match="changed during operation"):
        with proof_verification_scope():
            verify_persisted_proof(("authority", "digest"), verify)

    verify_persisted_proof(("authority", "digest"), verify)
    assert attempts == 3


def test_scope_reuses_loaded_value_and_rechecks_it_before_return() -> None:
    loads = 0
    checks: list[object] = []

    def load() -> object:
        nonlocal loads
        loads += 1
        return object()

    with proof_verification_scope():
        first = load_verified_value(("crop", "c1"), load, checks.append)
        second = load_verified_value(("crop", "c1"), load, checks.append)
        assert first is second
        assert loads == 1

    assert checks == [first]


def test_loaded_value_is_fresh_outside_scope() -> None:
    loads = 0

    def load() -> int:
        nonlocal loads
        loads += 1
        return loads

    assert load_verified_value(("crop", "c1"), load, lambda _value: None) == 1
    assert load_verified_value(("crop", "c1"), load, lambda _value: None) == 2


def test_finish_rechecks_before_commit_and_disables_later_reuse() -> None:
    loads = 0
    checks: list[int] = []

    def load() -> int:
        nonlocal loads
        loads += 1
        return loads

    with proof_verification_scope():
        assert load_verified_value(("crop", "c1"), load, checks.append) == 1
        assert load_verified_value(("crop", "c1"), load, checks.append) == 1
        finish_proof_verification()
        assert checks == [1]
        assert load_verified_value(("crop", "c1"), load, checks.append) == 2
        assert load_verified_value(("crop", "c1"), load, checks.append) == 3

    assert checks == [1]


def test_restart_opens_a_second_independently_closed_phase() -> None:
    calls: list[str] = []

    with proof_verification_scope():
        verify_persisted_proof(("phase", 1), lambda: calls.append("one"))
        finish_proof_verification()
        restart_proof_verification()
        verify_persisted_proof(("phase", 2), lambda: calls.append("two"))
        finish_proof_verification()

    assert calls == ["one", "one", "two", "two"]


def test_recursive_exact_proof_is_checked_once_per_phase() -> None:
    calls: list[str] = []

    def verify() -> None:
        calls.append("verify")
        verify_persisted_proof(("recursive", "digest"), verify)

    with proof_verification_scope():
        verify_persisted_proof(("recursive", "digest"), verify)

    assert calls == ["verify", "verify"]
