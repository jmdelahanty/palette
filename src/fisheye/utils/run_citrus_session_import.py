#!/usr/bin/env python3
"""Run the conservative Citrus session import workflow inside an LSF job.

This helper is intentionally import-only:

1. organize one completed Citrus session into the recordings store;
2. create/update analysis Zarrs from the organizer JSONL log;
3. optionally scan imported/skipped-existing Zarrs into a registry.

It does not run detect, refine, crops, keypoints, or masks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
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


def build_organize_command(
    *,
    session_dir: Path,
    dest_root: Path,
    log_dir: Path,
    apply: bool,
    rename_cams: bool,
    run_video_diagnostics: bool,
    run_h5_diagnostics: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "fisheye.utils.organize_recordings",
        str(session_dir),
        "--dest-root",
        str(dest_root),
        "--log-dir",
        str(log_dir),
        "--recursive",
        "--write-manifest",
        "--apply" if apply else "--dry-run",
    ]
    command.append("--rename-cams" if rename_cams else "--no-rename-cams")
    if run_video_diagnostics:
        command.append("--run-video-diagnostics")
    if run_h5_diagnostics:
        command.append("--run-h5-diagnostics")
    return command


def build_import_command(
    *,
    organize_log: Path,
    log_dir: Path,
    apply: bool,
    recording_only: bool,
    allow_preflight_failures: bool,
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
    if allow_preflight_failures:
        command.append("--allow-preflight-failures")
    if registry is not None:
        command.extend(["--registry", str(registry)])
    return command


def _run_command(command: Sequence[str], *, name: str, run_dir: Path) -> CommandRecord:
    stdout_path = run_dir / f"{name}.stdout.txt"
    stderr_path = run_dir / f"{name}.stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(list(command), stdout=stdout, stderr=stderr, check=False)
    return CommandRecord(
        name=name,
        command=list(command),
        returncode=int(result.returncode),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _newest_jsonl(log_dir: Path, pattern: str, *, before: set[Path]) -> Optional[Path]:
    candidates = sorted(set(log_dir.glob(pattern)) - before)
    if candidates:
        return candidates[-1]
    all_candidates = sorted(log_dir.glob(pattern))
    return all_candidates[-1] if all_candidates else None


def _read_zarr_paths_from_import_log(log_path: Optional[Path]) -> list[Path]:
    if log_path is None or not log_path.exists():
        return []
    paths: list[Path] = []
    seen: set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = payload.get("event")
        if event not in {"recording_plan", "recording_ok", "recording_skipped"}:
            continue
        status = payload.get("status")
        if status == "missing":
            continue
        zarr_path = payload.get("zarr_path")
        if not isinstance(zarr_path, str) or not zarr_path.strip():
            continue
        key = zarr_path.strip()
        if key not in seen:
            seen.add(key)
            paths.append(Path(key))
    return paths


def _read_recording_dirs_from_organize_log(log_path: Optional[Path]) -> list[Path]:
    if log_path is None or not log_path.exists():
        return []
    paths: list[Path] = []
    seen: set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "recording_applied":
            continue
        dest_dir = payload.get("dest_dir")
        if not isinstance(dest_dir, str) or not dest_dir.strip():
            continue
        key = dest_dir.strip()
        if key not in seen:
            seen.add(key)
            paths.append(Path(key))
    return paths


def _status_payload(
    *,
    args: argparse.Namespace,
    status: str,
    run_dir: Path,
    organize_log: Optional[Path],
    organized_recording_dirs: list[Path],
    import_log: Optional[Path],
    zarr_paths: list[Path],
    commands: list[CommandRecord],
    registry_file_list: Optional[Path],
) -> dict[str, object]:
    return {
        "schema_id": "palette.citrus_session_import.status.v1",
        "status": status,
        "session_dir": str(args.session_dir),
        "dest_root": str(args.dest_root),
        "run_dir": str(run_dir),
        "apply": bool(args.apply),
        "recording_only": bool(args.recording_only),
        "register": bool(args.register),
        "registry": str(args.registry) if args.registry is not None else None,
        "organize_log": str(organize_log) if organize_log is not None else None,
        "organized_recording_dirs": [str(path) for path in organized_recording_dirs],
        "import_log": str(import_log) if import_log is not None else None,
        "registry_file_list": str(registry_file_list) if registry_file_list is not None else None,
        "zarr_paths": [str(path) for path in zarr_paths],
        "commands": [asdict(command) for command in commands],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, help="Completed Citrus transfer session directory.")
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(os.environ.get("PALETTE_RECORDINGS_ROOT", DEFAULT_DEST_ROOT)),
        help=f"Organized recordings destination root (default: $PALETTE_RECORDINGS_ROOT or {DEFAULT_DEST_ROOT}).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Directory for workflow logs/status (default: <session-parent>/.processing_logs/citrus_import_manual_<timestamp>).",
    )
    parser.add_argument("--apply", action="store_true", help="Apply organization/import writes.")
    parser.add_argument("--dry-run", action="store_true", help="Plan organization/import without writes.")
    parser.add_argument("--no-rename-cams", action="store_true", help="Keep original camera filenames.")
    parser.add_argument("--recording-only", action="store_true", help="Import organized camera-video-only recordings without stimulus.")
    parser.add_argument("--allow-preflight-failures", action="store_true", help="Do not block import on manifest preflight failures.")
    parser.add_argument("--run-video-diagnostics", action="store_true", help="Run video diagnostics during organize apply.")
    parser.add_argument("--run-h5-diagnostics", action="store_true", help="Run H5 diagnostics during organize apply.")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Scan imported/skipped-existing analysis Zarrs into the registry during import.",
    )
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path used with --register.")
    parser.add_argument("--status-json", type=Path, help="Optional path for final status JSON.")

    args = parser.parse_args(argv)
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive.")
    if not args.apply:
        args.dry_run = True
    if args.register and args.registry is None:
        parser.error("--register requires --registry.")
    if not args.session_dir.exists():
        parser.error(f"session directory not found: {args.session_dir}")
    if not args.session_dir.is_dir():
        parser.error(f"session path is not a directory: {args.session_dir}")

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = (
            args.session_dir.parent
            / ".processing_logs"
            / f"citrus_import_manual_{_utc_timestamp_for_path()}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    organize_log_dir = run_dir / "organize_recordings"
    import_log_dir = run_dir / "import_organized_recordings_analysis"
    organize_log_dir.mkdir(parents=True, exist_ok=True)
    import_log_dir.mkdir(parents=True, exist_ok=True)

    commands: list[CommandRecord] = []
    organize_before = set(organize_log_dir.glob("organize_recordings_*.jsonl"))
    organize_command = build_organize_command(
        session_dir=args.session_dir,
        dest_root=args.dest_root,
        log_dir=organize_log_dir,
        apply=bool(args.apply),
        rename_cams=not bool(args.no_rename_cams),
        run_video_diagnostics=bool(args.run_video_diagnostics),
        run_h5_diagnostics=bool(args.run_h5_diagnostics),
    )
    print("Running organize command:")
    print(" ".join(organize_command))
    organize_result = _run_command(organize_command, name="01_organize_recordings", run_dir=run_dir)
    commands.append(organize_result)
    organize_log = _newest_jsonl(organize_log_dir, "organize_recordings_*.jsonl", before=organize_before)

    import_log: Optional[Path] = None
    organized_recording_dirs: list[Path] = []
    zarr_paths: list[Path] = []
    registry_file_list: Optional[Path] = None
    status = "ok"

    if organize_result.returncode != 0 or organize_log is None:
        status = "failed"
    else:
        organized_recording_dirs = _read_recording_dirs_from_organize_log(organize_log)
        if not organized_recording_dirs:
            print("Skipping import: organizer log has no recording_applied entries.")
        else:
            import_before = set(import_log_dir.glob("import_organized_recordings_analysis_*.jsonl"))
            import_command = build_import_command(
                organize_log=organize_log,
                log_dir=import_log_dir,
                apply=bool(args.apply),
                recording_only=bool(args.recording_only),
                allow_preflight_failures=bool(args.allow_preflight_failures),
                registry=args.registry if args.register else None,
            )
            print("Running import command:")
            print(" ".join(import_command))
            import_result = _run_command(import_command, name="02_import_organized_recordings_analysis", run_dir=run_dir)
            commands.append(import_result)
            import_log = _newest_jsonl(
                import_log_dir,
                "import_organized_recordings_analysis_*.jsonl",
                before=import_before,
            )
            zarr_paths = _read_zarr_paths_from_import_log(import_log)
            if import_result.returncode != 0:
                status = "failed"

    status_json = args.status_json or (run_dir / "citrus_session_import.status.json")
    payload = _status_payload(
        args=args,
        status=status,
        run_dir=run_dir,
        organize_log=organize_log,
        organized_recording_dirs=organized_recording_dirs,
        import_log=import_log,
        zarr_paths=zarr_paths,
        commands=commands,
        registry_file_list=registry_file_list,
    )
    status_json.parent.mkdir(parents=True, exist_ok=True)
    status_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status={status}")
    print(f"status_json={status_json}")
    if organize_log is not None:
        print(f"organize_log={organize_log}")
    if import_log is not None:
        print(f"import_log={import_log}")
    for path in zarr_paths:
        print(f"zarr_path={path}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
