"""Exit codes and the two outcomes intake distinguishes from ordinary failure.

The workflow runner (docs/design/2026-10-07-workflow-runner §5.3) maps these
codes without reading any other state:

- ``EXIT_DONE`` (0): published, and the matching probe is true;
- ``EXIT_REFUSED`` (65): the input is invalid (synthetic origin, admission-mode
  conflict, contract violation); an operator incident, never retried;
- ``EXIT_HELD`` (75): another live job holds the delivery's lock; "attached";
- ``EXIT_FAILED`` (1): any other failure; retried.
"""

from __future__ import annotations

from typing import Any, Mapping

EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_REFUSED = 65  # sysexits EX_DATAERR
EXIT_HELD = 75  # sysexits EX_TEMPFAIL


class IntakeRefused(ValueError):
    """Invalid input: retrying cannot help, an operator must look (exit 65).

    ``code`` is an optional machine-readable reason and ``details`` extra
    machine-readable fields; the CLI copies both into its error JSON.
    """

    def __init__(self, message: str, *, code: str | None = None, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class RegistrarCommitMismatch(IntakeRefused):
    """The registering checkout is not the commit that produced the receipts.

    Deterministic for this deployment: retrying from the same checkout can
    never succeed. Register from a deployment at ``receipt_producer_git_sha``.
    """


class IntakeHeld(RuntimeError):
    """Another live holder owns this delivery's intake lock (exit 75)."""

    def __init__(self, message: str, *, lock_path: str, holder: Mapping[str, Any] | None):
        super().__init__(message)
        self.lock_path = lock_path
        self.holder = dict(holder) if holder else None


# Organizer (``require``) invariant violations that are deterministic for a
# delivery: the same input fails the same way on every retry, so they are
# refusals (65). Ownership-lost and I/O failures are deliberately absent and
# stay retryable (1). Matched by message prefix because the organizer raises
# one exception type (TransferSnapshotError) for all of its invariants.
DETERMINISTIC_ORGANIZER_VIOLATIONS = (
    "parent manifest changed before retirement",
    "source inventory changed before retirement",
    "staging contains unplanned or reappeared source artifacts",
    "staging contains new unplanned directories",
    "staging contains a non-regular artifact",
    "completed staging source is no longer empty",
    "preparation cannot change its requested admission contract",
    "retirement cannot change its requested admission contract",
    "parent receipt has another source identity",
    "requested stimulus import is incomplete",
    "existing coordinator has another plan",
    "existing parent manifest differs from organization plan",
    "existing video sync assessment differs from the materialized videos",
    "completed parent receipts changed",
    "retirement parent receipts changed",
    "parent admission changed during retirement",
    "organization plan differs from live source",
)

# Per-parent import failures the import owner reports for its own
# deterministic preflight/contract refusals. Anything else (a child's
# traceback, an archive write failure) cannot be told apart from I/O and is
# retried.
DETERMINISTIC_IMPORT_STEPS = ("preflight_gate", "recording_import_preflight", "recording_import_sealed", "plan")


def classify_organizer_failure(exc: BaseException) -> BaseException:
    """Map a deterministic organizer invariant violation to IntakeRefused."""

    from fisheye.shared.recording_transfer_snapshot import TransferSnapshotError

    if isinstance(exc, TransferSnapshotError) and not isinstance(exc, IntakeRefused):
        message = str(exc)
        if "ownership lost" not in message and message.startswith(DETERMINISTIC_ORGANIZER_VIOLATIONS):
            refused = IntakeRefused(f"delivery violates an intake invariant: {message}", code="intake_invariant_violation")
            refused.__cause__ = exc
            return refused
    return exc


def exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, IntakeRefused):
        return EXIT_REFUSED
    if isinstance(exc, IntakeHeld):
        return EXIT_HELD
    return EXIT_FAILED


__all__ = [
    "EXIT_DONE",
    "EXIT_FAILED",
    "EXIT_HELD",
    "EXIT_REFUSED",
    "IntakeHeld",
    "IntakeRefused",
    "RegistrarCommitMismatch",
    "classify_organizer_failure",
    "exit_code_for",
]
