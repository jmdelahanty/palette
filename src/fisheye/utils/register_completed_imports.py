"""Register completed Citrus transfer imports from the designated writer host.

The canonical registry is one SQLite file on NFS, so exactly one host writes
it (``fisheye.registry.shadow_publish``). Cluster import jobs are dispatched
with ``--no-register`` (poller config ``"registration": "workstation"``); this
step runs on the writer host, normally from the same cron entry right after
``citrus_transfer_v2_poller``, and registers what those jobs imported.

For each delivery the poller marked submitted (``<state_dir>/<key>.submitted``)
and not yet registered, it reads the job's status JSON
(``<log_dir>/bsub_submissions/citrus_import_*_<key>/workflow-<job_id>/citrus_session_import.status.json``):

- no status and the job has not ended: still queued or running; next run;
- no status but the job ended (the launcher's ``*.<job_id>.status.txt`` or
  LSF's report in ``<job_id>.out`` exists): recorded as a failed import;
- status not ``complete``: the import failed; recorded once in
  ``<key>.import_failed`` for operator review, never retried here;
- complete: the delivery (the status's ``plan.snapshot_id``) is registered by
  ``fisheye.intake.register_delivery``: its import probe is re-verified and
  ALL of its Zarrs are synchronized in one registry publication, so a delivery
  is registered whole or not at all; ``<key>.registered`` records the result.
  A registration error is written to ``<key>.registration_failed`` and retried
  on the next run (registration is idempotent). A registration held by another
  live run is left for the next run.

Producer-declared synthetic transfers are never registered. Config (the
poller's JSON plus): ``registry``, ``writer_host`` (this host, short name or
FQDN, compared by ``fisheye.intake.registration.is_writer_host``),
``writer_lock_path``, ``shadow_temp_root``, ``shadow_backup_dir``; optional
``registration`` may only be ``"workstation"``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from fisheye.intake.delivery import refuse_synthetic_registration
from fisheye.intake.outcomes import IntakeHeld, IntakeRefused
from fisheye.intake.registration import RegistryWriter

REQUIRED = ("state_dir", "log_dir", "registry", "writer_host", "writer_lock_path",
            "shadow_temp_root", "shadow_backup_dir")


class RegistrarRefusal(ValueError):
    pass


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def load_config(path: Path) -> dict:
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise RegistrarRefusal(f"unreadable config {path}: {exc}") from exc
    missing = [key for key in REQUIRED if not (isinstance(config, dict) and config.get(key))]
    if missing:
        raise RegistrarRefusal(f"config missing required keys: {', '.join(missing)}")
    if config.get("registration", "workstation") != "workstation":
        raise RegistrarRefusal('config "registration" must be "workstation" (job mode is retired)')
    return config


def _writer(config: dict) -> RegistryWriter:
    try:
        return RegistryWriter.from_config(config)
    except IntakeRefused as exc:
        raise RegistrarRefusal(str(exc)) from exc


def _require_writer_host(config: dict) -> None:
    try:
        _writer(config).require_this_host()
    except IntakeRefused as exc:
        raise RegistrarRefusal(str(exc)) from exc


def _job_id(submitted: Path) -> str | None:
    ids = re.findall(r"^job_id=(\d+)\s*$", submitted.read_text(), flags=re.MULTILINE)
    return ids[-1] if ids else None


def _run_dir(config: dict, key: str, job_id: str) -> Path | None:
    """The launcher run directory that holds this job's outputs."""

    candidates = [
        path for path in (Path(config["log_dir"]) / "bsub_submissions").glob(f"citrus_import_*_{key}")
        if (path / f"workflow-{job_id}").exists() or (path / f"{job_id}.out").exists()
        or any(path.glob(f"*.{job_id}.status.txt"))
    ]
    if len(candidates) > 1:
        raise RegistrarRefusal(f"ambiguous run directory for key={key} job={job_id}: {candidates}")
    return candidates[0] if candidates else None


def _status_json(config: dict, key: str, job_id: str) -> Path | None:
    run_dir = _run_dir(config, key, job_id)
    if run_dir is None:
        return None
    path = run_dir / f"workflow-{job_id}" / "citrus_session_import.status.json"
    return path if path.is_file() else None


def _job_ended(config: dict, key: str, job_id: str) -> bool:
    """LSF appends its job report to -oo <job_id>.out; the job script writes status.txt last."""

    run_dir = _run_dir(config, key, job_id)
    if run_dir is None:
        return False
    if any(run_dir.glob(f"*.{job_id}.status.txt")):
        return True
    out = run_dir / f"{job_id}.out"
    return out.is_file() and "Resource usage summary" in out.read_text(errors="replace")


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


Register = Callable[[dict, str, Path], Mapping[str, str]]


def _default_register(config: dict, snapshot_sha: str, destination_root: Path) -> dict[str, str]:
    """One delivery through the intake owner; returns {zarr path: dataset id}."""

    from fisheye.intake.registration import register_delivery

    result = register_delivery(
        snapshot_sha, writer=_writer(config), destination_root=destination_root
    )
    return {binding["zarr_path"]: binding["dataset_id"] for binding in result.bindings or ()}


def register_completed(
    config: dict,
    *,
    dry_run: bool,
    register: Register = _default_register,
) -> int:
    state = Path(config["state_dir"])
    registry = Path(config["registry"])
    failures = 0
    for submitted in sorted(state.glob("*.submitted")):
        key = submitted.name[: -len(".submitted")]
        done = state / f"{key}.registered"
        import_failed = state / f"{key}.import_failed"
        if done.exists() or import_failed.exists():
            continue
        job_id = _job_id(submitted)
        if job_id is None:
            log(f"no job_id recorded for key={key}; skipping")
            continue
        status_path = _status_json(config, key, job_id)
        if status_path is None:
            if not _job_ended(config, key, job_id):
                log(f"pending: key={key} job={job_id} has no status yet")
                continue
            log(f"import failed: key={key} job={job_id} ended without a status JSON")
            if not dry_run:
                _write(import_failed, {"job_id": job_id, "status_json": None, "status": "failed",
                                       "error": "LSF job ended without writing its status JSON"})
            continue
        status: dict[str, Any] = json.loads(status_path.read_text())
        if status.get("status") != "complete" or not status.get("import_complete"):
            log(f"import failed: key={key} job={job_id}; recorded for operator review")
            if not dry_run:
                _write(import_failed, {"job_id": job_id, "status_json": str(status_path),
                                       "status": status.get("status"), "error": status.get("error")})
            continue
        zarr_paths = [Path(path) for path in status.get("zarr_paths") or []]
        if dry_run:
            log(f"dry-run: would register key={key} job={job_id} zarrs={[str(p) for p in zarr_paths]}")
            continue
        try:
            plan = status["plan"]
            refuse_synthetic_registration(plan, registry=registry)
            if not zarr_paths:
                raise RegistrarRefusal("complete import lists no zarr_paths")
            datasets = dict(
                register(config, str(plan["snapshot_id"]), Path(plan["destination_root"]))
            )
        except IntakeHeld as exc:
            log(f"pending: key={key} job={job_id}: {exc}")
            continue
        except Exception as exc:  # retried on the next run
            failures += 1
            log(f"registration failed: key={key} job={job_id}: {exc}")
            _write(state / f"{key}.registration_failed",
                   {"job_id": job_id, "error": str(exc),
                    "failed_utc": datetime.now(timezone.utc).isoformat()})
            continue
        _write(done, {"job_id": job_id, "status_json": str(status_path), "registry": str(registry),
                      "datasets": datasets, "registered_utc": datetime.now(timezone.utc).isoformat()})
        (state / f"{key}.registration_failed").unlink(missing_ok=True)
        log(f"registered key={key} job={job_id} datasets={sorted(datasets.values())}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=os.environ.get("CITRUS_V2_POLLER_CONFIG"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("--config or CITRUS_V2_POLLER_CONFIG is required")
    try:
        config = load_config(args.config)
        if args.dry_run:
            return register_completed(config, dry_run=True)
        _require_writer_host(config)
        with open(Path(config["state_dir"]) / "registrar.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log("another registrar run is in progress; exiting")
                return 0
            return register_completed(config, dry_run=False)
    except RegistrarRefusal as exc:
        log(f"refusing: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
