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
from typing import Any, ParamSpec, TypeVar, cast


P = ParamSpec("P")
R = TypeVar("R")
ProofVerifier = Callable[[], None]


@dataclass
class _ProofVerificationSession:
    """Verified proof callbacks owned by one outermost operation."""

    _verified: dict[Hashable, ProofVerifier] = field(default_factory=dict)
    _values: dict[Hashable, Any] = field(default_factory=dict)
    _closing: bool = False
    _closing_callbacks: dict[Hashable, ProofVerifier] = field(default_factory=dict)
    _closing_verified: set[Hashable] = field(default_factory=set)
    _verifying: set[Hashable] = field(default_factory=set)
    _finished: bool = False

    def verify(self, key: Hashable, verifier: ProofVerifier) -> None:
        if self._finished:
            verifier()
            return
        if self._closing:
            if key in self._closing_verified or key in self._verifying:
                return
            callback = self._closing_callbacks.get(key, verifier)
            self._verifying.add(key)
            try:
                callback()
                self._closing_verified.add(key)
            finally:
                self._verifying.remove(key)
            return
        if key in self._verified:
            return
        if key in self._verifying:
            # The outer verifier is currently establishing this exact proof.
            return
        self._verifying.add(key)
        try:
            verifier()
            # Register only after the persisted proof succeeds.
            self._verified[key] = verifier
        finally:
            self._verifying.remove(key)

    def load_value(
        self,
        key: Hashable,
        loader: Callable[[], R],
        verifier: Callable[[R], None],
    ) -> R:
        if self._closing or self._finished:
            return loader()
        if key in self._values:
            return cast(R, self._values[key])
        value = loader()
        self._values[key] = value
        self._verified.setdefault(key, lambda: verifier(value))
        return value

    def reverify(self) -> None:
        self._closing = True
        self._closing_callbacks = dict(self._verified)
        self._closing_verified.clear()
        try:
            for key, verifier in tuple(self._closing_callbacks.items()):
                self.verify(key, verifier)
        finally:
            self._closing = False
            self._closing_callbacks.clear()
            self._closing_verified.clear()
            self._verifying.clear()

    def finish(self) -> None:
        """Perform the closing recheck and disable reuse for the remaining scope."""

        if self._finished:
            return
        try:
            self.reverify()
        finally:
            self._verified.clear()
            self._values.clear()
            self._finished = True

    def restart(self) -> None:
        if not self._finished:
            raise RuntimeError(
                "Cannot restart proof verification before the current phase finishes."
            )
        self._finished = False


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
        if not session._finished:
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


def load_verified_value(
    key: Hashable,
    loader: Callable[[], R],
    verifier: Callable[[R], None],
) -> R:
    """Load once within an operation and freshly verify before publication.

    Outside a scope this is a normal fresh load. Within a scope the exact key
    reuses the bound value, while ``verifier`` must reload and compare the
    persisted authority during the closing recheck.
    """

    session = _ACTIVE_PROOF_SESSION.get()
    if session is None:
        return loader()
    return session.load_value(key, loader, verifier)


def finish_proof_verification() -> None:
    """Close the active proof phase before an operation commits output.

    Subsequent verification calls in the surrounding scope run fresh and are
    not retained. This lets a writer recheck immutable inputs while its
    fail-closed publication boundary is still active, rather than discovering
    drift after selectors have been promoted.
    """

    session = _ACTIVE_PROOF_SESSION.get()
    if session is not None:
        session.finish()


def restart_proof_verification() -> None:
    """Begin another verification phase inside the same writer operation."""

    session = _ACTIVE_PROOF_SESSION.get()
    if session is not None:
        session.restart()


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
    "finish_proof_verification",
    "load_verified_value",
    "restart_proof_verification",
    "verify_persisted_proof",
]
