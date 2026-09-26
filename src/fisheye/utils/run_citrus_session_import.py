#!/usr/bin/env python3
"""Ingest one completed Citrus transfer (transfer-v2) inside an LSF job.

Transfer-v2 parent intake is the only way new recordings enter Palette
(``citrus_transfer_parent_workflow``): it organizes exact parent recordings,
imports their analysis Zarrs (routing unified H5s to the native profile), can
register them, and retires verified staging. It does not run detect, refine,
crops, keypoints, or masks. The legacy per-H5 organizer poller was removed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_DEST_ROOT = Path("/groups/johnson/johnsonlab/jeremy/recordings")


@dataclass(frozen=True)
class CommandRecord:
    name: str
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str


def _utc_timestamp_for_path() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_import_command(
    *,
    organize_log: Path,
    log_dir: Path,
    apply: bool,
    recording_only: bool,
    registry: Optional[Path],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "fisheye.utils.import_organized_recordings_analysis",
        "--organize-log",
        str(organize_log),
        "--log-dir",
        str(log_dir),
        "--apply" if apply else "--dry-run",
    ]
    if recording_only:
        command.append("--recording-only")
    if registry is not None:
        command.extend(["--registry", str(registry)])
    return command


def _run_command(command: Sequence[str], *, name: str, run_dir: Path,
                 pass_fds: tuple[int, ...] = (), env: dict[str, str] | None = None) -> CommandRecord:
    stdout_path = run_dir / f"{name}.stdout.txt"
    stderr_path = run_dir / f"{name}.stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        # Opt-in intake lends its lease to the actual writer, so SIGKILL of
        # this supervisor cannot admit another concurrent writer. Legacy calls
        # keep their original subprocess arguments and behavior.
        result = subprocess.run(list(command), stdout=stdout, stderr=stderr, check=False,
                                **({"pass_fds": pass_fds} if pass_fds else {}),
                                **({"env": env} if env is not None else {}))
    return CommandRecord(
        name=name,
        command=list(command),
        returncode=int(result.returncode),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _newest_jsonl(log_dir: Path, pattern: str, *, before: set[Path]) -> Optional[Path]:
    candidates = sorted(set(log_dir.glob(pattern)) - before)
    # A previous invocation or ambiguous concurrent logs cannot acknowledge
    # this command. Fresh logs are required even for an idempotent replay.
    return candidates[0] if len(candidates) == 1 else None


def _read_zarr_paths_from_import_log(log_path: Optional[Path]) -> list[Path]:
    if log_path is None or not log_path.exists():
        return []
    paths: list[Path] = []
    seen: set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("import log entry must be a JSON object")
        event = payload.get("event")
        if event in {"recording_failed", "registry_sync_failed"}:
            raise ValueError(f"import log contains {event}")
        if event not in {"recording_ok", "recording_skipped"}:
            continue
        status = payload.get("status")
        if status == "missing":
            raise ValueError("import log contains a missing recording")
        if event == "recording_skipped" and status != "skipped":
            raise ValueError("import log has an invalid skipped-recording acknowledgment")
        zarr_path = payload.get("zarr_path")
        if not isinstance(zarr_path, str) or not zarr_path.strip():
            raise ValueError("import acknowledgment has no zarr_path")
        key = zarr_path.strip()
        if key not in seen:
            seen.add(key)
            paths.append(Path(key))
    return paths


def _verify_import_acknowledgments(
    *,
    import_log: Path | None,
    recording_dirs: Sequence[Path],
    zarr_paths: Sequence[Path],
    recording_only: bool,
) -> None:
    """Bind this invocation's acknowledgments to every planned live artifact."""

    from fisheye.utils.import_recording_analysis import resolve_single_recording_plan
    from fisheye.utils.import_organized_recordings_analysis import _existing_analysis_complete

    if import_log is None:
        raise ValueError("importer produced no fresh unambiguous JSONL log")
    expected = {
        resolve_single_recording_plan(
            recording_dir=path, require_h5=not recording_only,
        ).zarr_path.resolve()
        for path in recording_dirs
    }
    acknowledged = {path.resolve() for path in zarr_paths}
    if not expected or acknowledged != expected:
        raise ValueError("import acknowledgments do not exactly cover organized recording outputs")
    for path in sorted(expected):
        complete, reason = _existing_analysis_complete(path, require_stimulus=not recording_only)
        if not complete:
            raise ValueError(f"acknowledged import is not validated: {path}: {reason}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, help="Completed Citrus transfer session directory.")
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(os.environ.get("PALETTE_RECORDINGS_ROOT", DEFAULT_DEST_ROOT)),
        help=f"Organized recordings destination root (default: $PALETTE_RECORDINGS_ROOT or {DEFAULT_DEST_ROOT}).",
    )
    parser.add_argument("--run-dir", type=Path, help="Directory for workflow logs/status.")
    parser.add_argument("--apply", action="store_true", help="Apply organization/import writes.")
    parser.add_argument("--dry-run", action="store_true", help="Plan organization/import without writes.")
    parser.add_argument("--register", action="store_true", help="Scan imported analysis Zarrs into the registry.")
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path used with --register.")
    parser.add_argument("--status-json", type=Path, help="Optional path for final status JSON.")
    parser.add_argument("--resume-transfer-plan", type=Path, help="Exact saved organization plan for retry, including interrupted retirement.")

    args = parser.parse_args(argv)
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive.")
    if not args.apply:
        args.dry_run = True
    if args.register and args.registry is None:
        parser.error("--register requires --registry.")
    from fisheye.utils.citrus_transfer_parent_workflow import run_transfer_parent_workflow

    return run_transfer_parent_workflow(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
