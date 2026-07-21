from __future__ import annotations

import pytest

from fisheye.shared.proof_verification import (
    proof_verification_operation,
    proof_verification_scope,
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
