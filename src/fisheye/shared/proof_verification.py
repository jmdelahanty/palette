"""Operation-scoped reuse of exact persisted-provenance verification.

Coordinate readers deliberately re-open persisted authorities instead of
trusting an in-memory object.  A single high-level publication can encounter
the same digest-bound authority thousands of times while traversing its
provenance graph, however.  Repeating the same synchronous metadata reads at
every edge adds latency without establishing a different fact.

This module permits one successful proof to be reused only inside one explicit
operation scope.  The outermost scope re-runs every distinct verifier before it
returns successfully.  A later operation therefore starts empty and reads the
store again, while a mutation observed by the closing recheck still fails the
current operation.

There is intentionally no process-global or time-based cache.  Callers must
key proofs by the validated object's identity plus exact archive identity,
persisted paths, and content digests.  Scopes are appropriate only for
authorities whose mutation is prohibited during the operation; a transient
mutation that is completely restored before the closing check is not
observable.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")
ProofVerifier = Callable[[], None]


@dataclass
class _ProofVerificationSession:
    """Verified proof callbacks owned by one outermost operation."""

    _verified: dict[Hashable, ProofVerifier] = field(default_factory=dict)
    _closing: bool = False

    def verify(self, key: Hashable, verifier: ProofVerifier) -> None:
        if self._closing:
            # A closing verifier must not silently introduce an unchecked
            # dependency after the final-pass snapshot has been taken.
            verifier()
            return
        if key in self._verified:
            return
        verifier()
        # Register only after the persisted proof succeeds.
        self._verified[key] = verifier

    def reverify(self) -> None:
        self._closing = True
        try:
            for verifier in tuple(self._verified.values()):
                verifier()
        finally:
            self._closing = False


_ACTIVE_PROOF_SESSION: ContextVar[_ProofVerificationSession | None] = ContextVar(
    "palette_active_proof_verification_session",
    default=None,
)


@contextmanager
def proof_verification_scope() -> Iterator[None]:
    """Reuse exact proofs within one operation and recheck them at its end.

    Nested scopes join the current operation.  Only the outermost scope owns
    the closing recheck and clears the cache.
    """

    current = _ACTIVE_PROOF_SESSION.get()
    if current is not None:
        yield
        return

    session = _ProofVerificationSession()
    token = _ACTIVE_PROOF_SESSION.set(session)
    try:
        yield
        session.reverify()
    finally:
        _ACTIVE_PROOF_SESSION.reset(token)


def verify_persisted_proof(
    key: Hashable,
    verifier: ProofVerifier,
) -> None:
    """Verify now, or reuse the exact proof in the active operation scope."""

    session = _ACTIVE_PROOF_SESSION.get()
    if session is None:
        verifier()
        return
    session.verify(key, verifier)


def proof_verification_operation(
    function: Callable[P, R],
) -> Callable[P, R]:
    """Run one synchronous public operation in a fresh-or-nested proof scope."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with proof_verification_scope():
            return function(*args, **kwargs)

    return wrapped


__all__ = [
    "proof_verification_operation",
    "proof_verification_scope",
    "verify_persisted_proof",
]
