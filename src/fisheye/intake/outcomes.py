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
    """Invalid input: retrying cannot help, an operator must look (exit 65)."""


class IntakeHeld(RuntimeError):
    """Another live holder owns this delivery's intake lock (exit 75)."""

    def __init__(self, message: str, *, lock_path: str, holder: Mapping[str, Any] | None):
        super().__init__(message)
        self.lock_path = lock_path
        self.holder = dict(holder) if holder else None


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
    "exit_code_for",
]
