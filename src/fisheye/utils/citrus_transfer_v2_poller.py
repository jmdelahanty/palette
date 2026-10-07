"""Poll Citrus staging for completed transfer-v2 sessions and submit intake.

Tracked replacement for the untracked ``~/bin/citrus_staging_marker_poller.sh``
(v1). A session is submitted only when its ``_citrus_transfer_complete.json``
declares ``citrus.transfer_completion_marker.v2`` and binds the snapshot bytes
under ``_citrus_transfer/snapshot.json`` by sha256. v1 markers are logged and
never submitted. Full inventory verification stays in the LSF job
(``verify_transfer_snapshot``); this poller only decides what to submit.

Recording context is the producer's per-camera declaration in the snapshot,
so the operator JSON config (``--config`` or ``CITRUS_V2_POLLER_CONFIG``) holds
only operations: ``staging_dir``, ``state_dir``, ``log_dir``, ``submit`` =
``{"transport": "local"|"ssh", "host": ..., "repo": ...}`` and optionally
``registration``, which may only be ``"workstation"``: jobs always run
``--no-register`` and ``register_completed_imports`` registers from the
writer host (``fisheye.intake.register_delivery``). Job-mode registration is
retired, so a config declaring ``"registration": "job"`` is refused rather
than silently reinterpreted. A config that still declares recording context
is refused. Marker validation is ``fisheye.intake.discovery.check_marker``.

Installation (cron entry, config file, retiring the v1 poller) is a separate,
user-authorized step. This module installs nothing and edits no crontab; run
``--dry-run`` first to see exactly what would be submitted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable

from fisheye.intake.discovery import (
    LEGACY_MARKER_SCHEMA,
    MarkerRefusal as PollerRefusal,
    check_marker,
)
from fisheye.shared.recording_transfer_snapshot import MARKER_NAME, SNAPSHOT_PATH

LAUNCHER = "scripts/submit_citrus_session_import_bsub.sh"
Runner = Callable[[list[str]], subprocess.CompletedProcess]


def log(message: str) -> None:
    print(message, flush=True)


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PollerRefusal(f"unreadable config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise PollerRefusal("config must be a JSON object")
    missing = [
        k for k in ("staging_dir", "state_dir", "log_dir", "submit") if not config.get(k)
    ]
    if missing:
        raise PollerRefusal(f"config missing required keys: {', '.join(missing)}")
    stale = sorted(
        {"recording_type", "recording_subtype", "behavior_mode", "recording_only"}
        & set(config)
    )
    if stale:
        raise PollerRefusal(
            f"recording context comes from the producer snapshot; remove {', '.join(stale)}"
        )
    submit = config["submit"]
    if (
        not isinstance(submit, dict)
        or submit.get("transport") not in ("local", "ssh")
        or not submit.get("repo")
    ):
        raise PollerRefusal("submit needs transport local|ssh and repo")
    if submit["transport"] == "ssh" and not submit.get("host"):
        raise PollerRefusal("ssh transport needs submit.host")
    if config.get("registration", "workstation") != "workstation":
        raise PollerRefusal(
            'job-mode registration is retired; registration must be "workstation" '
            "(the writer host registers with register_completed_imports)"
        )
    return config


def claim_key(marker_path: Path, marker_sha256: str) -> str:
    """Path + marker content: rewritten markers never reuse an old claim."""

    return hashlib.sha256(f"{marker_path.resolve()}\0{marker_sha256}".encode()).hexdigest()


def build_command(config: dict, session_dir: Path, key: str) -> list[str]:
    launcher = [
        LAUNCHER,
        "--session-dir", str(session_dir),
        "--marker-key", key,
        "--log-dir", str(Path(config["log_dir"]) / "bsub_submissions"),
    ]
    # The job imports only; register_completed_imports registers from the
    # designated writer host after the job completes. --no-register is a
    # no-op for the current launcher but keeps an older checkout at
    # submit.repo (whose default was to register) from registering.
    launcher.append("--no-register")
    submit = config["submit"]
    remote = f"cd {shlex.quote(submit['repo'])} && {shlex.join(launcher)}"
    if submit["transport"] == "ssh":
        return ["ssh", submit["host"], remote]
    return ["bash", "-c", remote]


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def poll(config: dict, *, dry_run: bool, runner: Runner = _run) -> int:
    staging = Path(config["staging_dir"])
    state = Path(config["state_dir"])
    if not staging.is_dir():
        raise PollerRefusal(f"staging directory does not exist: {staging}")
    failures = 0
    for marker_path in sorted(staging.rglob(MARKER_NAME)):
        session_dir = marker_path.parent
        try:
            marker = check_marker(marker_path)
        except (PollerRefusal, OSError) as exc:
            log(f"refused marker={marker_path}: {exc}")
            continue
        if marker is None:
            log(f"legacy v1 marker ignored: {marker_path}")
            continue
        key = claim_key(marker_path, marker["_marker_sha256"])
        command = build_command(config, session_dir, key)
        claimed, submitted = state / f"{key}.claimed", state / f"{key}.submitted"
        if submitted.exists():
            continue
        if claimed.exists():
            # Polls run one at a time under poller.lock, so a claim without a
            # submission belongs to an earlier poll that died before recording
            # it. Retry: the launcher returns the earlier job if LSF has one.
            log(f"retrying unfinished claim key={key}")
        if dry_run:
            log(f"dry-run: would submit {shlex.join(command)}")
            continue
        fd = os.open(claimed, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        with os.fdopen(fd, "w") as handle:
            json.dump({"marker": str(marker_path), "marker_sha256": marker["_marker_sha256"],
                       "snapshot_id": marker["snapshot_id"], "command": command}, handle)
        log(f"submitting {shlex.join(command)}")
        result = runner(command)
        if result.returncode == 0:
            submitted.write_text(result.stdout or "")
            log(f"submitted key={key}")
        else:
            claimed.unlink()
            (state / f"{key}.failed").write_text((result.stdout or "") + (result.stderr or ""))
            log(f"submission failed rc={result.returncode}; will retry key={key}")
            failures += 1
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=os.environ.get("CITRUS_V2_POLLER_CONFIG"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("--config or CITRUS_V2_POLLER_CONFIG is required")
    try:
        config = load_config(Path(args.config))
        if args.dry_run:
            return poll(config, dry_run=True)
        state = Path(config["state_dir"])
        state.mkdir(parents=True, exist_ok=True)
        with open(state / "poller.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log("another poller invocation is running; exiting")
                return 0
            return poll(config, dry_run=False)
    except PollerRefusal as exc:
        log(f"refusing to poll: {exc}")
        return 2


__all__ = [
    "LEGACY_MARKER_SCHEMA",
    "MARKER_NAME",
    "PollerRefusal",
    "SNAPSHOT_PATH",
    "build_command",
    "check_marker",
    "claim_key",
    "load_config",
    "main",
    "poll",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
