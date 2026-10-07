"""One delivery's durable intake evidence, admission mode and claim locks.

A delivery is named by its snapshot sha (the hex digest in the transfer
marker's ``snapshot_id``). Its durable record is the organizer's
``<destination_root>/.transfer_intake/<sha>/organization_state.json``; this
module only reads it. The organizer stays the owner of that state, of the
plan, and of the lock primitive (``_coordinator_lock``) both claims use.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Callable, Iterator, Mapping

from fisheye.intake.outcomes import IntakeHeld, IntakeRefused
from fisheye.shared.recording_transfer_snapshot import strict_json

CANONICAL_REGISTRY = Path(
    "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
)
DEFAULT_DESTINATION_ROOT = Path("/groups/johnson/johnsonlab/jeremy/recordings")
STATE_DIRECTORY = ".transfer_intake"
STATE_FILE = "organization_state.json"
IMPORT_LOCK_KIND = ".workflow.lock"  # organize_transfer_recordings.transfer_parent_workflow_lock
REGISTER_LOCK_KIND = ".register.lock"
DURABLE_STATES = ("reserved", "materialized", "retiring", "complete")
_SHA = re.compile(r"[0-9a-f]{64}")


def default_destination_root() -> Path:
    return Path(os.environ.get("PALETTE_RECORDINGS_ROOT", DEFAULT_DESTINATION_ROOT))


def validate_snapshot_sha(snapshot_sha: str) -> str:
    """The bare lowercase hex digest; ``sha256:`` prefixes are accepted."""

    value = str(snapshot_sha).removeprefix("sha256:")
    if not _SHA.fullmatch(value):
        raise IntakeRefused(f"not a snapshot sha256: {snapshot_sha!r}")
    return value


def state_directory(destination_root: Path, snapshot_sha: str) -> Path:
    return Path(destination_root) / STATE_DIRECTORY / validate_snapshot_sha(snapshot_sha)


def load_durable_state(destination_root: Path, snapshot_sha: str) -> dict | None:
    """The organizer's recovery state, or None before the first reservation."""

    path = state_directory(destination_root, snapshot_sha) / STATE_FILE
    if not path.exists() and not path.is_symlink():
        return None
    try:
        state = strict_json(path)
    except Exception as exc:
        raise IntakeRefused(f"unreadable intake state {path}: {exc}") from exc
    plan = state.get("plan")
    if not isinstance(plan, dict) or plan.get("snapshot_id") != f"sha256:{validate_snapshot_sha(snapshot_sha)}":
        raise IntakeRefused(f"intake state {path} does not hold this delivery's plan")
    return state


def admission_mode(state: Mapping[str, Any] | None) -> str:
    """How the delivery's first attempt asked to be admitted.

    ``workstation``: the job imports only and the writer host registers;
    ``job``: the retired mode in which the LSF job registered
    (``registry_path`` recorded); ``unset``: no attempt has recorded one yet.
    The stored value is reported as is, never reinterpreted.
    """

    contract = (state or {}).get("admission_contract")
    if contract is None:
        return "unset"
    if isinstance(contract, Mapping) and contract.get("registry_path") is None:
        return "workstation"
    return "job"


def require_workstation_admission(state: Mapping[str, Any] | None, *, require_stimulus: bool) -> None:
    """Refuse to resume a delivery under any contract but the one it started with."""

    mode = admission_mode(state)
    if mode == "job":
        raise IntakeRefused(
            "delivery was started under the retired job-mode registration "
            f"(admission_contract={state['admission_contract']!r}); it is reported, "
            "not resumed: resolve it manually"
        )
    contract = (state or {}).get("admission_contract")
    if contract is not None and contract != {"registry_path": None, "require_stimulus": require_stimulus}:
        raise IntakeRefused(
            f"delivery admission contract {contract!r} conflicts with this attempt "
            f"(require_stimulus={require_stimulus})"
        )


def plan_recording_only(plan: Mapping[str, Any]) -> bool:
    """Stimulus import follows the producer's declared intent, never H5 presence."""

    intents = {parent["context"]["recording_intent"] for parent in plan["parents"]}
    if len(intents) != 1:
        raise IntakeRefused(
            f"one transfer must declare one recording intent, got {sorted(intents)}"
        )
    return intents == {"recording_only"}


def plan_is_synthetic(plan: Mapping[str, Any]) -> bool:
    return any(parent["context"]["data_origin"] == "synthetic" for parent in plan["parents"])


def refuse_synthetic_registration(
    plan: Mapping[str, Any], *, registry: Path, allow_synthetic: bool = False
) -> None:
    """Producer-declared synthetic data never registers by default.

    ``allow_synthetic`` admits it only into an isolated (non-canonical)
    registry, for fixtures and canaries; it can never open the canonical one.
    """

    if not plan_is_synthetic(plan):
        return
    if not allow_synthetic:
        raise IntakeRefused("producer-declared synthetic transfer is never registered")
    if Path(registry).resolve() == CANONICAL_REGISTRY.resolve():
        raise IntakeRefused("synthetic transfer cannot register into the canonical registry")


def _holder_record() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
    }


def _read_holder(lock_path: Path) -> dict[str, Any] | None:
    try:
        with open(lock_path, "rb") as handle:
            holder = json.loads(handle.read(4096) or b"null")
    except (OSError, ValueError):
        return None
    return holder if isinstance(holder, dict) else None


@contextmanager
def claim(plan: Mapping[str, Any], *, kind: str) -> Iterator[tuple[Callable[[], None], int]]:
    """Take one of intake's two delivery locks, or raise :class:`IntakeHeld`.

    Both are the organizer's ``fcntl.flock(LOCK_EX | LOCK_NB)`` coordinator
    lock beside the state directory; the import claim is exactly
    ``transfer_parent_workflow_lock``. Callers take it before any side
    effect. The holder's host/pid/LSF job are written into the lock file for
    the status view; the content is informational, the flock is the claim.
    """

    from fisheye.utils.organize_transfer_recordings import (
        _coordinator_lock,
        _state_directory,
        transfer_parent_workflow_lock,
    )

    lock_path = _state_directory(plan).with_suffix(kind)
    manager = (
        transfer_parent_workflow_lock(plan)
        if kind == IMPORT_LOCK_KIND
        else _coordinator_lock(plan, kind=kind)
    )
    with ExitStack() as stack:
        try:
            verify, descriptor = stack.enter_context(manager)
        except BlockingIOError as exc:
            raise IntakeHeld(
                f"delivery {plan['snapshot_id']} is held by another live job",
                lock_path=str(lock_path),
                holder=_read_holder(lock_path),
            ) from exc
        # Replace the previous holder's record. pwrite at offset 0 never moves
        # the shared descriptor offset (and lands at 0 even under O_APPEND,
        # since the file is empty after the truncate).
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, json.dumps(_holder_record(), sort_keys=True).encode(), 0)
        yield verify, descriptor


__all__ = [
    "CANONICAL_REGISTRY",
    "DEFAULT_DESTINATION_ROOT",
    "DURABLE_STATES",
    "IMPORT_LOCK_KIND",
    "REGISTER_LOCK_KIND",
    "admission_mode",
    "claim",
    "default_destination_root",
    "load_durable_state",
    "plan_is_synthetic",
    "plan_recording_only",
    "refuse_synthetic_registration",
    "require_workstation_admission",
    "state_directory",
    "validate_snapshot_sha",
]
