"""Register completed Citrus transfer imports from the designated writer host.

The canonical registry is one SQLite file on NFS, so exactly one host writes
it (``fisheye.registry.shadow_publish``). Cluster import jobs are dispatched
with ``--no-register`` (poller config ``"registration": "workstation"``); this
step runs on the writer host, normally from the same cron entry right after
``citrus_transfer_v2_poller``, and registers what those jobs imported.

For each delivery the poller marked submitted (``<state_dir>/<key>.submitted``)
and not yet registered, it reads the job's status JSON
(``<log_dir>/bsub_submissions/citrus_import_*_<key>/*.<job_id>.status.json``):

- no status yet: the job is still queued or running; try again next run;
- status not ``complete``: the import failed; recorded once in
  ``<key>.import_failed`` for operator review, never retried here;
- complete: every imported Zarr's immutable import receipt is verified
  (``load_verified_recording_import_receipt``) and the Zarr is registered
  through ``shadow_synchronize_recording_import``; ``<key>.registered`` records
  the result. A registration error is written to ``<key>.registration_failed``
  and retried on the next run (registration is an idempotent upsert).

Producer-declared synthetic transfers are never registered into the canonical
registry. Config (the poller's JSON plus): ``registry``, ``writer_host`` (must
equal this host's name), ``writer_lock_path``, ``shadow_temp_root``,
``shadow_backup_dir``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

REQUIRED = ("state_dir", "log_dir", "registry", "writer_host", "writer_lock_path",
            "shadow_temp_root", "shadow_backup_dir")
DECIDED_BY = "fisheye.utils.register_completed_imports"


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
    if config.get("registration") != "workstation":
        raise RegistrarRefusal('config "registration" must be "workstation" for this step')
    return config


def _writer_environment(config: dict) -> None:
    """The single-writer gateway reads its configuration from the environment."""

    if socket.gethostname() != config["writer_host"]:
        raise RegistrarRefusal(
            f"this host {socket.gethostname()!r} is not the writer {config['writer_host']!r}"
        )
    os.environ.update({
        "PALETTE_REGISTRY_WRITER_HOST": config["writer_host"],
        "PALETTE_REGISTRY_WRITER_LOCK_PATH": config["writer_lock_path"],
        "PALETTE_REGISTRY_SHADOW_TEMP_ROOT": config["shadow_temp_root"],
        "PALETTE_REGISTRY_SHADOW_BACKUP_DIR": config["shadow_backup_dir"],
    })


def _job_id(submitted: Path) -> str | None:
    ids = re.findall(r"^job_id=(\d+)\s*$", submitted.read_text(), flags=re.MULTILINE)
    return ids[-1] if ids else None


def _status_json(config: dict, key: str, job_id: str) -> Path | None:
    found = sorted(
        (Path(config["log_dir"]) / "bsub_submissions").glob(
            f"citrus_import_*_{key}/*.{job_id}.status.json"
        )
    )
    if len(found) > 1:
        raise RegistrarRefusal(f"ambiguous status for key={key} job={job_id}: {found}")
    return found[0] if found else None


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _default_register(registry: Path, zarr_path: Path) -> str:
    from fisheye.registry.recording_identity_authority import load_verified_recording_import_receipt
    from fisheye.registry.shadow_publish import shadow_synchronize_recording_import

    receipt = load_verified_recording_import_receipt(zarr_path)
    publication = shadow_synchronize_recording_import(
        canonical_registry=registry, zarr_path=zarr_path, receipt=receipt, decided_by=DECIDED_BY,
    )
    return str(publication.mutation_result["dataset_id"])


def register_completed(
    config: dict,
    *,
    dry_run: bool,
    register: Callable[[Path, Path], str] = _default_register,
) -> int:
    from fisheye.utils.citrus_transfer_parent_workflow import (
        _refuse_synthetic_canonical_registration,
    )

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
            log(f"pending: key={key} job={job_id} has no status yet")
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
            _refuse_synthetic_canonical_registration(
                status["plan"], SimpleNamespace(register=True, registry=registry)
            )
            if not zarr_paths:
                raise RegistrarRefusal("complete import lists no zarr_paths")
            datasets = {str(path): register(registry, path) for path in zarr_paths}
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
        _writer_environment(config)
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
