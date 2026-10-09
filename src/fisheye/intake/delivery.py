"""One delivery's durable intake evidence, admission mode and claim locks.

A delivery is named by its snapshot sha (the hex digest in the transfer
marker's ``snapshot_id``). Its durable record is the organizer's
``<destination_root>/.transfer_intake/<sha>/organization_state.json``; this
module only reads it. The organizer stays the owner of that state, of the
plan, and of the lock primitive (``_coordinator_lock``) both claims use.

Job-mode deliveries (operator path)
-----------------------------------
A delivery whose first attempt recorded ``admission_contract.registry_path``
(the retired "the LSF job registers" mode) is reported by ``discover``
(``legacy_mode``) and refused (65) by ``import_delivery`` and
``register_delivery``: intake never resumes it in another mode. Finish it in
its recorded mode by hand, on the registry writer host, with the
single-writer environment set (``PALETTE_REGISTRY_WRITER_HOST``,
``PALETTE_REGISTRY_WRITER_LOCK_PATH``, ``PALETTE_REGISTRY_SHADOW_TEMP_ROOT``,
``PALETTE_REGISTRY_SHADOW_BACKUP_DIR``) and from a deployment at the commit
that will produce the receipts:

1. read ``plan = state["plan"]`` and ``contract = state["admission_contract"]``
   from ``<destination_root>/.transfer_intake/<sha>/organization_state.json``;
2. for each ``parent["destination_dir"]`` of the plan, run the organizer's
   import owner in the recorded mode::

       scripts/py -m fisheye.utils.import_organized_recordings_analysis \
           <destination_dir> --apply --registry <contract registry_path> \
           [--recording-only   # only when contract require_stimulus is false]

3. retire with the exact recorded contract::

       scripts/py -c 'import json, sys; from pathlib import Path
       from fisheye.utils.organize_transfer_recordings import finalize_transfer_staging
       state = json.loads(Path(sys.argv[1]).read_text()); c = state["admission_contract"]
       finalize_transfer_staging(state["plan"], registry_path=Path(c["registry_path"]),
                                 require_stimulus=c["require_stimulus"])' <organization_state.json>

``finalize_transfer_staging`` re-verifies every receipt and the registry
admission before retiring; a delivery it refuses needs investigation, not a
mode change.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable, Iterator, Mapping

from fisheye.intake.outcomes import IntakeHeld, IntakeRefused, IntakeTransient
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


# The per-recording archives on /nvme1 were deliberately deleted (2026-09-03);
# /groups is the only live recordings store.
RETIRED_STORE_ROOT = Path("/nvme1")


def require_live_destination_root(path: Path | str) -> Path:
    """Refuse a destination root on the retired /nvme1 store."""

    root = Path(path)
    absolute = Path(os.path.abspath(root))
    if absolute == RETIRED_STORE_ROOT or RETIRED_STORE_ROOT in absolute.parents:
        raise IntakeRefused(
            f"destination root {root} is on the retired /nvme1 store; the live "
            f"recordings store is {DEFAULT_DESTINATION_ROOT} (check PALETTE_RECORDINGS_ROOT)",
            code="retired_destination_root",
        )
    return root


def default_destination_root() -> Path:
    return require_live_destination_root(
        os.environ.get("PALETTE_RECORDINGS_ROOT", DEFAULT_DESTINATION_ROOT)
    )


def validate_snapshot_sha(snapshot_sha: str) -> str:
    """The bare lowercase hex digest; ``sha256:`` prefixes are accepted."""

    value = str(snapshot_sha).removeprefix("sha256:")
    if not _SHA.fullmatch(value):
        raise IntakeRefused(f"not a snapshot sha256: {snapshot_sha!r}")
    return value


def state_directory(destination_root: Path, snapshot_sha: str) -> Path:
    return Path(destination_root) / STATE_DIRECTORY / validate_snapshot_sha(snapshot_sha)


STATE_READ_ATTEMPTS = 5
STATE_READ_BACKOFF_S = 0.2
CHANGED_WHILE_READ = "changed while read"  # source_recording_identity.load_strict_json_object


def _read_state(path: Path) -> dict:
    """Read the organizer's state, riding out a concurrent atomic save.

    Another attempt holding the delivery's lock may save the state while it
    is read; the strict loader then reports the document "changed while
    read". That is retried a few times and, if it persists, raised as a
    retryable :class:`IntakeTransient` (1), never as a malformed-state
    refusal (65). A read failure with an OSError cause is I/O (1).
    """

    for attempt in range(STATE_READ_ATTEMPTS):
        try:
            return strict_json(path)
        except OSError:
            raise  # transient I/O: retryable (1), never a refusal
        except Exception as exc:
            # The shared strict loader wraps a read failure in its own
            # ValueError; an I/O cause is still I/O, not a malformed state.
            cause = exc.__cause__
            while cause is not None and not isinstance(cause, OSError):
                cause = cause.__cause__
            if isinstance(cause, OSError):
                raise cause from exc
            if CHANGED_WHILE_READ not in str(exc):
                raise IntakeRefused(f"malformed intake state {path}: {exc}") from exc
            if attempt + 1 == STATE_READ_ATTEMPTS:
                raise IntakeTransient(
                    f"intake state {path} kept changing while read (a concurrent save); retry"
                ) from exc
            time.sleep(STATE_READ_BACKOFF_S * (attempt + 1))
    raise AssertionError("unreachable")


def load_durable_state(destination_root: Path, snapshot_sha: str) -> dict | None:
    """The organizer's recovery state, or None before the first reservation."""

    path = state_directory(destination_root, snapshot_sha) / STATE_FILE
    if not path.exists() and not path.is_symlink():
        return None
    state = _read_state(path)
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
            "not resumed: finish it in its recorded mode by hand, see "
            "'Job-mode deliveries (operator path)' in fisheye.intake.delivery",
            code="legacy_job_mode_delivery",
            details={"admission_contract": state["admission_contract"]},
        )
    contract = (state or {}).get("admission_contract")
    if contract is not None and contract != {"registry_path": None, "require_stimulus": require_stimulus}:
        raise IntakeRefused(
            f"delivery admission contract {contract!r} conflicts with this attempt "
            f"(require_stimulus={require_stimulus})",
            code="admission_contract_conflict",
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
    if same_file(Path(registry), CANONICAL_REGISTRY):
        raise IntakeRefused("synthetic transfer cannot register into the canonical registry")


def same_file(left: Path, right: Path) -> bool:
    """File identity by (st_dev, st_ino); by resolved path when either is absent.

    A bind mount, hard link or second automount path to the canonical
    registry is still the canonical registry.
    """

    try:
        a, b = os.stat(left), os.stat(right)
    except FileNotFoundError:
        return Path(left).resolve() == Path(right).resolve()
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def find_delivery_by_source(
    destination_root: Path, source_dir: Path
) -> tuple[str | None, list[dict[str, str]]]:
    """The snapshot sha whose durable plan names ``source_dir``, if any.

    Lets a caller holding only a (possibly already retired) staging directory
    find its delivery without the marker. Unrelated states that cannot be
    read or are malformed are skipped and returned for reporting; they never
    fail the lookup. Only an ambiguous match (two plans name this source)
    refuses.
    """

    root = Path(destination_root) / STATE_DIRECTORY
    skipped: list[dict[str, str]] = []
    if not root.is_dir():
        return None, skipped
    wanted = str(Path(source_dir).absolute().resolve())
    found = []
    for state_file in sorted(root.glob(f"*/{STATE_FILE}")):
        if not _SHA.fullmatch(state_file.parent.name):
            continue
        try:
            state = load_durable_state(destination_root, state_file.parent.name)
        except (OSError, IntakeRefused, IntakeTransient) as exc:
            skipped.append({"path": str(state_file), "reason": str(exc)})
            continue
        if state is not None and state["plan"].get("source_dir") == wanted:
            found.append(state_file.parent.name)
    if len(found) > 1:
        raise IntakeRefused(f"several intake deliveries name staging source {wanted}: {found}")
    return (found[0] if found else None), skipped


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
    "require_live_destination_root",
    "find_delivery_by_source",
    "load_durable_state",
    "plan_is_synthetic",
    "plan_recording_only",
    "refuse_synthetic_registration",
    "require_workstation_admission",
    "same_file",
    "state_directory",
    "validate_snapshot_sha",
]
