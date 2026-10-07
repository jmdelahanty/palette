"""``import_delivery``: organize, import and retire one transfer-v2 delivery.

This is the LSF side of intake. It never writes the registry: the delivery
is admitted in workstation mode (``admission_contract.registry_path`` is
None) and ``register_delivery`` registers it on the writer host.

Order of operations (the claim comes before any side effect):

1. resolve the exact plan without writing anything: the durable
   ``state["plan"]`` when the organizer has one, else an explicit resume plan,
   else a fresh plan from the live sealed marker;
2. validate the attempt's run directory and status path (both must be new);
3. take the delivery's workflow lock (``transfer_parent_workflow_lock``),
   or raise :class:`IntakeHeld` having created nothing;
4. create the run directory and reserve the status file;
5. replay retirement (``retiring``/``complete``) or prepare the parents,
   import them in-process through the recording import owner and retire.

The parent list is handed straight to the import owner
(``import_organized_recordings_analysis.build_plans`` and
``import_recording_analysis.process_recording_import``); there is no
subprocess, synthetic organize log or importer-log parsing. The one remaining
child, the stimulus importer, inherits the lock descriptor through
``PALETTE_RECORDING_IMPORT_LEASE_FD`` so the lease outlives a killed
supervisor while that child still writes.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator, Mapping

from fisheye.intake.delivery import (
    IMPORT_LOCK_KIND,
    claim,
    default_destination_root,
    load_durable_state,
    plan_recording_only,
    require_workstation_admission,
    validate_snapshot_sha,
)
from fisheye.intake.outcomes import (
    DETERMINISTIC_IMPORT_STEPS,
    IntakeRefused,
    classify_organizer_failure,
)
from fisheye.intake.probes import ProbeResult, import_probe_result
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    TRANSFER_PARENT_LAYOUTS,
    TransferSnapshotError,
    strict_json,
)

STATUS_SCHEMA = "palette.citrus_transfer_parent_intake.status.v1"
STATUS_NAME = "citrus_session_import.status.json"
PLAN_NAME = "organization_plan.json"
LEASE_FD_ENV = "PALETTE_RECORDING_IMPORT_LEASE_FD"


class _ImportFailed(RuntimeError):
    """One or more parents did not import; retried (exit 1)."""


class _PlanRefusal(ValueError):
    """Intake's own explicit refusal of one organized parent (deterministic)."""


@dataclass(frozen=True)
class ParentImportResult:
    recording_id: str
    camera_id: str
    recording_dir: str
    zarr_path: str
    outcome: str  # "imported", "already_imported" or "failed"
    failed_step: str | None = None
    error: str | None = None
    returncode: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@contextmanager
def _lease_environment(descriptor: int) -> Iterator[None]:
    """Lend the workflow lock to the stimulus child the import owner spawns."""

    previous = os.environ.get(LEASE_FD_ENV)
    os.environ[LEASE_FD_ENV] = str(descriptor)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LEASE_FD_ENV, None)
        else:
            os.environ[LEASE_FD_ENV] = previous


def import_parents(
    plan: Mapping[str, Any], *, recording_only: bool, lease_fd: int
) -> list[ParentImportResult]:
    """Import every planned parent through the recording import owner.

    ``build_plans`` resolves each organized parent and marks one whose
    receipt already verifies as ``skipped`` (resume); every other parent goes
    through ``process_recording_import`` in this process. Every planned
    parent is attempted; the structured per-parent results are returned and
    the caller decides. The resolved Zarr must be the organizer's planned one.
    """

    from fisheye.utils.import_organized_recordings_analysis import build_plans
    from fisheye.utils.import_recording_analysis import (
        RecordingAnalysisPlan,
        RecordingImportOptions,
        process_recording_import,
    )
    from fisheye.utils.organize_transfer_recordings import parent_zarr_paths

    import_stimulus = not recording_only
    options = RecordingImportOptions(
        import_video_metadata=True,
        video_metadata_overwrite=False,
        import_stimulus=import_stimulus,
        stimulus_always=False,
        stimulus_run_name=None,
        stimulus_overwrite=False,
        stimulus_quiet=False,
    )
    results = []
    with _lease_environment(lease_fd):
        for parent, planned_zarr in zip(plan["parents"], parent_zarr_paths(plan)):
            directory = Path(parent["destination_dir"])
            base = {
                "recording_id": parent["identity"]["recording_id"],
                "camera_id": parent["identity"]["camera_id"],
                "recording_dir": str(directory),
                "zarr_path": str(planned_zarr),
            }
            try:
                [resolved] = build_plans(
                    [directory],
                    import_stimulus=import_stimulus,
                    skip_existing=True,
                    check_stimulus=import_stimulus,
                )
                if resolved.zarr_path.resolve() != planned_zarr.resolve():
                    raise _PlanRefusal(
                        f"import owner resolved {resolved.zarr_path}, planned {planned_zarr}"
                    )
                if resolved.status == "missing":
                    raise _PlanRefusal(f"organized parent is not importable: {resolved.reason}")
                if resolved.status == "skipped":
                    results.append(ParentImportResult(**base, outcome="already_imported"))
                    continue
                result = process_recording_import(
                    RecordingAnalysisPlan(
                        recording_dir=resolved.recording_dir,
                        h5_path=resolved.h5_path,
                        cam_video=resolved.cam_video,
                        zarr_path=resolved.zarr_path,
                        recording_layout=resolved.recording_layout,
                    ),
                    options,
                )
            except _PlanRefusal as exc:  # deterministic: refused
                results.append(
                    ParentImportResult(**base, outcome="failed", failed_step="plan_refused", error=str(exc))
                )
                continue
            except Exception as exc:  # I/O, zarr/h5py, a code bug: retried
                results.append(
                    ParentImportResult(
                        **base, outcome="failed", failed_step="plan_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if result.ok:
                results.append(ParentImportResult(**base, outcome="imported"))
            else:
                results.append(
                    ParentImportResult(
                        **base,
                        outcome="failed",
                        failed_step=result.failed_step,
                        returncode=result.returncode,
                        error=result.error
                        or (f"returncode={result.returncode}" if result.returncode is not None else None),
                    )
                )
    return results


def _deterministic(parent: ParentImportResult) -> bool:
    """A refusal the import owner will repeat on every retry of this input.

    The owner's own preflight and contract steps, and its stimulus refusals
    (exit 2 from ``run_stimulus_import``: no H5, unsupported unified profile,
    missing finalization receipt). A stimulus *child* that dies (exit 1)
    cannot be told apart from I/O and is retried.
    """

    if parent.failed_step in ("recording_import_preflight", "recording_import_sealed"):
        # These steps also wrap read failures of the existing archive and the
        # checkout identity; only a failure that names neither is a refusal.
        if "[Errno" in (parent.error or ""):
            return False
        if CLEAN_CHECKOUT_REFUSAL in (parent.error or ""):
            return _checkout_identity_is_readable()
    if parent.failed_step in DETERMINISTIC_IMPORT_STEPS:
        return True
    return parent.failed_step == "import_stimulus_to_zarr" and parent.returncode == 2


# import_recording_analysis._clean_importer_code_identity's refusal. It is
# raised both for a dirty checkout (deterministic for this deployment) and
# when git itself failed (transient); intake tells them apart by asking git.
CLEAN_CHECKOUT_REFUSAL = "current recording imports require a clean commit-pinned Palette checkout"


def _importer_checkout() -> Mapping[str, Any]:
    from fisheye.shared import run_provenance
    import fisheye.utils.import_recording_analysis as importer

    return run_provenance.git_identity(cwd=Path(importer.__file__).resolve().parents[3])


def _checkout_identity_is_readable() -> bool:
    code = _importer_checkout()
    return code.get("git_sha") is not None and code.get("git_dirty") is not None


def _require_one_producer_commit(plan: Mapping[str, Any]) -> None:
    """Refuse to mix producer commits within one delivery (before importing).

    Registration binds a receipt only from its producing checkout, so a
    delivery whose parents were imported by different commits can never be
    registered by any one deployment. If a parent already carries a receipt
    from another commit than this checkout, refuse before importing anything
    else or retiring staging: a retry from the original deployment finishes.
    """

    from fisheye.intake.probes import receipt_producer_git_sha
    from fisheye.shared.recording_import_receipt import recording_import_receipt_paths
    from fisheye.utils.organize_transfer_recordings import parent_zarr_paths

    existing = {}
    for zarr_path in parent_zarr_paths(plan):
        if not zarr_path.exists():
            continue
        for receipt_path in recording_import_receipt_paths(zarr_path):
            existing[str(zarr_path)] = receipt_producer_git_sha(zarr_path, receipt_path.stem)
    if not existing:
        return
    code = _importer_checkout()
    current, dirty = code.get("git_sha"), code.get("git_dirty")
    if current is None or dirty is None:
        raise RuntimeError("the importer checkout's git identity is unavailable; retry")
    foreign = {path: sha for path, sha in existing.items() if sha != current}
    if foreign:
        raise IntakeRefused(
            "parents of this delivery were already imported by another commit "
            f"({sorted(set(foreign.values()))}) than this checkout ({current}); finish the "
            "delivery from a deployment at that commit",
            code="delivery_producer_commit_mismatch",
            details={
                "receipt_producer_git_shas": sorted(set(foreign.values())),
                "importer_git_sha": current,
                "zarr_paths": sorted(foreign),
            },
        )


def _find_session(staging_dir: Path, snapshot_sha: str) -> Path | None:
    from fisheye.intake.discovery import MarkerRefusal, check_marker

    for marker_path in sorted(Path(staging_dir).rglob(MARKER_NAME)):
        try:
            marker = check_marker(marker_path)
        except (MarkerRefusal, OSError):
            continue
        if marker is not None and marker["snapshot_id"] == f"sha256:{snapshot_sha}":
            return marker_path.parent
    return None


def _resolve_plan(
    snapshot_sha: str,
    *,
    destination_root: Path,
    resume_plan: Path | None,
    session_dir: Path | None,
    staging_dir: Path | None,
) -> tuple[dict, dict | None]:
    """The exact plan for this attempt, read without any write."""

    from fisheye.utils.organize_transfer_recordings import (
        _validate_plan,
        build_transfer_organization_plan,
    )

    state = load_durable_state(destination_root, snapshot_sha)
    explicit = None
    if resume_plan is not None:
        explicit = strict_json(Path(resume_plan))
        _validate_plan(explicit, live_source=False)
    if state is not None:
        plan = state["plan"]
        if explicit is not None and explicit != plan:
            raise IntakeRefused("resume plan differs from the delivery's durable plan")
    elif explicit is not None:
        plan = explicit
    else:
        source = session_dir
        if source is None and staging_dir is not None:
            source = _find_session(staging_dir, snapshot_sha)
        if source is None:
            raise IntakeRefused(
                "delivery has no durable intake state and no sealed marker was given or found"
            )
        plan = build_transfer_organization_plan(
            Path(source).absolute(), destination_root=destination_root
        )
    if plan["snapshot_id"] != f"sha256:{snapshot_sha}":
        raise IntakeRefused(f"plan is for {plan['snapshot_id']}, not sha256:{snapshot_sha}")
    if plan["destination_root"] != str(Path(destination_root).resolve()):
        raise IntakeRefused("plan names another destination root")
    if session_dir is not None and plan["source_dir"] != str(Path(session_dir).absolute().resolve()):
        raise IntakeRefused("plan names another staging source")
    if plan["recording_layout"] not in TRANSFER_PARENT_LAYOUTS:
        raise IntakeRefused("parent workflow supports rolling_clips or single_video only")
    return plan, state


def _validated_outputs(plan: Mapping[str, Any], run_dir: Path, status_path: Path | None) -> tuple[Path, Path]:
    """Check (never create) the attempt's run directory and status file."""

    from fisheye.utils.organize_transfer_recordings import (
        _separate_destination,
        _state_directory,
    )

    source = Path(plan["source_dir"])
    run_dir = _separate_destination(source.resolve(), Path(run_dir))
    for parent in plan["parents"]:
        _separate_destination(Path(parent["destination_dir"]), run_dir)
    state_parent = _state_directory(plan).parent
    _separate_destination(state_parent, run_dir)
    if run_dir.exists():
        raise IntakeRefused(f"run directory already exists: {run_dir}")
    selected = Path(status_path or run_dir / STATUS_NAME).absolute()
    if any(path.is_symlink() for path in (selected, *selected.parents)):
        raise IntakeRefused("status path contains a symlink")
    selected = selected.resolve()
    for protected in (
        source,
        state_parent,
        *(Path(p["destination_dir"]) for p in plan["parents"]),
    ):
        if selected.is_relative_to(protected):
            raise IntakeRefused("status path would mutate protected intake evidence")
    if selected.exists():
        raise IntakeRefused("status destination already exists")
    if selected.is_relative_to(run_dir) and selected != run_dir / STATUS_NAME:
        raise IntakeRefused("custom status path overlaps workflow outputs")
    return run_dir, selected


class _Status:
    """The attempt's status JSON: reserved exclusively, rewritten in place."""

    def __init__(self, path: Path, payload: dict):
        self.path = path
        self.payload = payload
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            reserved = os.fstat(descriptor)
            self.identity = (reserved.st_dev, reserved.st_ino)
        finally:
            os.close(descriptor)

    def publish(self) -> None:
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise RuntimeError("status destination contains a symlink")
        current = self.path.lstat()
        if not (stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == self.identity):
            raise RuntimeError("status destination ownership lost")
        write_json_atomic(self.path, self.payload)
        current = self.path.lstat()
        self.identity = (current.st_dev, current.st_ino)


def import_delivery(
    snapshot_sha: str,
    run_dir: Path,
    resume_plan: Path | None = None,
    *,
    destination_root: Path | None = None,
    session_dir: Path | None = None,
    staging_dir: Path | None = None,
    status_path: Path | None = None,
) -> ProbeResult:
    """Organize, import and retire one delivery; return its true import probe.

    Idempotent and resumable from every durable state, including
    ``retiring`` with the marker already gone. ``run_dir`` must not exist.
    Raises :class:`IntakeRefused` (65) for invalid input, :class:`IntakeHeld`
    (75) when another live job holds the delivery, anything else (1) for a
    retryable failure. No registry is opened.
    """

    sha = validate_snapshot_sha(snapshot_sha)
    destination = Path(destination_root) if destination_root is not None else default_destination_root()
    try:
        plan, state = _resolve_plan(
            sha,
            destination_root=destination,
            resume_plan=resume_plan,
            session_dir=session_dir,
            staging_dir=staging_dir,
        )
        recording_only = plan_recording_only(plan)
        require_workstation_admission(state, require_stimulus=not recording_only)
        run_dir, selected_status = _validated_outputs(plan, Path(run_dir), status_path)
    except (TransferSnapshotError, KeyError, TypeError) as exc:
        raise IntakeRefused(f"invalid delivery input: {exc}") from exc

    payload: dict[str, Any] = {
        "schema_id": STATUS_SCHEMA,
        "status": "failed",
        "import_complete": False,
        "staging_finalized": False,
        "plan": plan,
        "apply": True,
        "registry": None,
        "snapshot_sha": sha,
    }
    status: _Status | None = None
    with ExitStack() as resources:
        verify_lock, lock_fd = resources.enter_context(claim(plan, kind=IMPORT_LOCK_KIND))
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            status = _Status(selected_status, payload)
            plan_path = run_dir / PLAN_NAME
            write_json_atomic(plan_path, plan, overwrite=False)
            payload.update(run_dir=str(run_dir), organization_plan_path=str(plan_path))
            try:
                result = _import_under_lock(
                    sha, plan, recording_only=recording_only, verify_lock=verify_lock,
                    lock_fd=lock_fd, payload=payload,
                )
            except Exception as exc:
                classified = classify_organizer_failure(exc)
                if classified is exc:
                    raise
                raise classified from exc
            payload.update(status="complete", probe_import=result.to_json())
            status.publish()
            return result
        except BaseException as exc:
            payload["error"] = str(exc)
            if status is not None:
                try:
                    status.publish()
                except Exception as status_exc:
                    print(f"Status not written: {status_exc}", file=sys.stderr)
            raise


def _import_under_lock(
    sha: str,
    plan: dict,
    *,
    recording_only: bool,
    verify_lock,
    lock_fd: int,
    payload: dict,
) -> ProbeResult:
    from fisheye.utils.organize_transfer_recordings import (
        finalize_transfer_staging,
        parent_zarr_paths,
        prepare_transfer_parent_recordings,
    )

    destination = Path(plan["destination_root"])
    state = load_durable_state(destination, sha)  # re-read under the claim
    if state is not None and state["plan"] != plan:
        raise IntakeRefused("durable intake state holds another plan")
    require_workstation_admission(state, require_stimulus=not recording_only)
    if state is not None and state.get("status") in {"retiring", "complete"}:
        # Staging may already be retired: replay the exact journal, never
        # rebuild the plan or re-import an immutable publication.
        final = finalize_transfer_staging(
            plan, registry_path=None, require_stimulus=not recording_only
        )
    else:
        prepare_transfer_parent_recordings(
            plan, registry_path=None, require_stimulus=not recording_only
        )
        _require_one_producer_commit(plan)
        parents = import_parents(plan, recording_only=recording_only, lease_fd=lock_fd)
        payload["parents"] = [parent.to_json() for parent in parents]
        failed = [parent for parent in parents if parent.outcome == "failed"]
        if failed:
            message = "parent import failed: " + "; ".join(
                f"{p.recording_dir}: {p.failed_step}: {p.error}" for p in failed
            )
            if all(_deterministic(parent) for parent in failed):
                raise IntakeRefused(message, code="parent_import_refused")
            raise _ImportFailed(message)
        payload["import_complete"] = True
        verify_lock()
        final = finalize_transfer_staging(
            plan, registry_path=None, require_stimulus=not recording_only
        )
    payload.update(
        zarr_paths=[str(path) for path in parent_zarr_paths(plan)],
        import_complete=True,
        staging_finalized=True,
        import_receipts=final["import_receipts"],
        retired_file_count=len(final["retired_files"]),
    )
    return import_probe_result(sha, plan=plan, receipts=final["import_receipts"], state=final)


__all__ = [
    "LEASE_FD_ENV",
    "ParentImportResult",
    "STATUS_NAME",
    "STATUS_SCHEMA",
    "import_delivery",
    "import_parents",
]
