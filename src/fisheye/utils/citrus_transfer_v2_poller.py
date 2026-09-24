"""Poll Citrus staging for completed transfer-v2 sessions and submit intake.

Tracked replacement for the untracked ``~/bin/citrus_staging_marker_poller.sh``
(v1). A session is submitted only when its ``_citrus_transfer_complete.json``
declares ``citrus.transfer_completion_marker.v2`` and binds the snapshot bytes
under ``_citrus_transfer/snapshot.json`` by sha256. v1 markers are logged and
never submitted. Full inventory verification stays in the LSF job
(``verify_transfer_snapshot``); this poller only decides what to submit.

Recording context comes only from the operator JSON config (``--config`` or
``CITRUS_V2_POLLER_CONFIG``); nothing is inferred. Keys: ``staging_dir``,
``state_dir``, ``log_dir``, ``recording_type``, ``recording_subtype``,
``behavior_mode``, optional ``recording_only`` (bool) and ``writer_host``,
and ``submit`` = ``{"transport": "local"|"ssh", "host": ..., "repo": ...}``.

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

from fisheye.shared.recording_manifest_context import recording_manifest_context_issues
from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    MARKER_SCHEMA,
    SNAPSHOT_PATH,
)

LEGACY_MARKER_SCHEMA = "citrus.transfer_completion_marker.v1"
LAUNCHER = "scripts/submit_citrus_session_import_bsub.sh"
Runner = Callable[[list[str]], subprocess.CompletedProcess]


class PollerRefusal(ValueError):
    pass


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
        k for k in ("staging_dir", "state_dir", "log_dir", "recording_type",
                    "recording_subtype", "behavior_mode", "submit")
        if not config.get(k)
    ]
    if missing:
        raise PollerRefusal(f"config missing required keys: {', '.join(missing)}")
    issues = recording_manifest_context_issues(
        {**config, "artifact_schema_id": "orange_transfer_parent_v1"}
    )
    if issues:
        raise PollerRefusal(f"invalid recording context: {issues}")
    if type(config.get("recording_only", False)) is not bool:
        raise PollerRefusal("recording_only must be a JSON boolean")
    submit = config["submit"]
    if (
        not isinstance(submit, dict)
        or submit.get("transport") not in ("local", "ssh")
        or not submit.get("repo")
    ):
        raise PollerRefusal("submit needs transport local|ssh and repo")
    if submit["transport"] == "ssh" and not submit.get("host"):
        raise PollerRefusal("ssh transport needs submit.host")
    return config


def check_marker(marker_path: Path) -> dict | None:
    """Return the v2 marker, None for a legacy v1 marker; raise if malformed."""

    marker_bytes = marker_path.read_bytes()
    try:
        marker = json.loads(marker_bytes)
    except ValueError as exc:
        raise PollerRefusal(f"invalid JSON: {exc}") from exc
    if not isinstance(marker, dict):
        raise PollerRefusal("marker is not a JSON object")
    if marker.get("schema_id") == LEGACY_MARKER_SCHEMA:
        return None
    snapshot = marker.get("snapshot")
    if (
        marker.get("schema_id") != MARKER_SCHEMA
        or marker.get("schema_version") != 2
        or marker.get("status") != "transfer_complete"
        or marker.get("required_consumer_profile") != "parent_recording_intake_v1"
        or not isinstance(snapshot, dict)
        or snapshot.get("path") != SNAPSHOT_PATH
        or marker.get("recording_payload_kind") not in ("citrus_h5", "video_only")
    ):
        raise PollerRefusal(f"not a complete transfer-v2 marker (schema_id={marker.get('schema_id')!r})")
    snapshot_file = marker_path.parent / SNAPSHOT_PATH
    if not snapshot_file.is_file():
        raise PollerRefusal("snapshot file missing")
    digest = hashlib.sha256(snapshot_file.read_bytes()).hexdigest()
    if digest != snapshot.get("sha256") or marker.get("snapshot_id") != f"sha256:{digest}":
        raise PollerRefusal("snapshot bytes do not match marker binding")
    marker["_marker_sha256"] = hashlib.sha256(marker_bytes).hexdigest()
    return marker


def claim_key(marker_path: Path, marker_sha256: str) -> str:
    """Path + marker content: rewritten markers never reuse an old claim."""

    return hashlib.sha256(f"{marker_path.resolve()}\0{marker_sha256}".encode()).hexdigest()


def build_command(config: dict, session_dir: Path, key: str) -> list[str]:
    launcher = [
        LAUNCHER,
        "--session-dir", str(session_dir),
        "--marker-key", key,
        "--log-dir", str(Path(config["log_dir"]) / "bsub_submissions"),
        "--recording-type", config["recording_type"],
        "--recording-subtype", config["recording_subtype"],
        "--behavior-mode", config["behavior_mode"],
    ]
    if config.get("recording_only"):
        launcher.append("--recording-only")
    if config.get("writer_host"):
        launcher += ["--writer-host", config["writer_host"]]
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
        if (marker["recording_payload_kind"] == "video_only") != bool(config.get("recording_only")):
            log(f"refused marker={marker_path}: payload {marker['recording_payload_kind']} "
                f"disagrees with config recording_only={config.get('recording_only', False)}")
            continue
        key = claim_key(marker_path, marker["_marker_sha256"])
        command = build_command(config, session_dir, key)
        claimed, submitted = state / f"{key}.claimed", state / f"{key}.submitted"
        if claimed.exists() or submitted.exists():
            continue
        if dry_run:
            log(f"dry-run: would submit {shlex.join(command)}")
            continue
        try:
            fd = os.open(claimed, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            continue
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
