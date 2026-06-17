#!/usr/bin/env python3
"""Finalize staging batches after organize/import verification.

The command is dry-run by default. With ``--apply`` it moves verified staging
batches into ``.processed/`` unless ``--delete`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from fisheye.shared.batch_logging import JsonLogger, make_run_id


DEFAULT_STAGING_ROOT = Path("/nvme1/staging")
DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


@dataclass(frozen=True)
class ImportLogStatus:
    ok: frozenset[str] = frozenset()
    failed: Mapping[str, str] = field(default_factory=dict)
    skipped: Mapping[str, str] = field(default_factory=dict)


@dataclass
class FinalizePlan:
    batch_path: Path
    recordings: list[Path]
    status: str
    action: str
    target_path: Optional[Path] = None
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest_paths: list[Path] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    missing_zarrs: list[Path] = field(default_factory=list)
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("batch_path", "target_path"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        for key in ("recordings", "manifest_paths", "missing_zarrs"):
            payload[key] = [str(path) for path in payload[key]]
        return payload


def _resolve_default_staging_root() -> Path:
    return Path(os.environ.get("PALETTE_STAGING_ROOT", str(DEFAULT_STAGING_ROOT))).expanduser()


def _resolve_default_recordings_root() -> Path:
    return Path(os.environ.get("PALETTE_RECORDINGS_ROOT", str(DEFAULT_RECORDINGS_ROOT))).expanduser()


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except OSError:
        return path.expanduser().absolute()


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        _safe_resolve(path).relative_to(_safe_resolve(parent))
        return True
    except ValueError:
        return False


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _iter_manifest_paths(recordings_root: Path) -> Iterable[Path]:
    if recordings_root.is_file() and recordings_root.name == "recording_manifest.json":
        yield recordings_root
        return
    if (recordings_root / "recording_manifest.json").is_file():
        yield recordings_root / "recording_manifest.json"
        return
    yield from recordings_root.rglob("recording_manifest.json")


def _manifest_source_matches_batch(manifest: Mapping[str, Any], batch_path: Path) -> bool:
    source_dir = manifest.get("source_dir")
    if not isinstance(source_dir, str) or not source_dir.strip():
        return False
    source = Path(source_dir).expanduser()
    batch = batch_path.expanduser()
    return _safe_resolve(source) == _safe_resolve(batch) or _path_is_relative_to(source, batch)


def _recordings_from_manifests(batch_path: Path, recordings_root: Path) -> list[Path]:
    recordings: set[Path] = set()
    for manifest_path in _iter_manifest_paths(recordings_root):
        payload = _load_json(manifest_path)
        if payload is None:
            continue
        if _manifest_source_matches_batch(payload, batch_path):
            recordings.add(manifest_path.parent)
    return sorted(recordings)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _recordings_from_organize_logs(
    batch_path: Path,
    organize_logs: Iterable[Path],
) -> tuple[list[Path], list[str]]:
    recordings: set[Path] = set()
    warnings: list[str] = []
    target = _safe_resolve(batch_path)
    for log_path in organize_logs:
        current_batch: Optional[Path] = None
        for payload in _read_jsonl(log_path):
            event = payload.get("event")
            if event == "batch_start":
                batch_source = payload.get("batch_source")
                current_batch = Path(batch_source).expanduser() if isinstance(batch_source, str) else None
                continue
            if current_batch is not None and _safe_resolve(current_batch) != target:
                continue
            if event == "recording_applied":
                dest_dir = payload.get("dest_dir")
                if isinstance(dest_dir, str) and dest_dir.strip():
                    recordings.add(Path(dest_dir).expanduser())
            elif event == "warning":
                message = payload.get("message")
                if isinstance(message, str) and message.strip():
                    warnings.append(f"{log_path}: {message}")
    return sorted(recordings), warnings


def read_import_log_status(import_logs: Iterable[Path]) -> ImportLogStatus:
    ok: set[str] = set()
    failed: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for log_path in import_logs:
        for payload in _read_jsonl(log_path):
            event = payload.get("event")
            recording_dir = payload.get("recording_dir")
            if not isinstance(recording_dir, str) or not recording_dir.strip():
                continue
            key = str(_safe_resolve(Path(recording_dir)))
            if event == "recording_ok":
                ok.add(key)
            elif event == "recording_failed":
                failed[key] = str(payload.get("error") or payload.get("step") or "recording_failed")
            elif event == "recording_skipped":
                skipped[key] = str(payload.get("reason") or payload.get("status") or "recording_skipped")
    return ImportLogStatus(frozenset(ok), failed, skipped)


def _manifest_file_paths(recording_dir: Path, manifest: Mapping[str, Any]) -> list[Path]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return []
    paths: list[Path] = []
    for section in ("raw", "cams", "derived"):
        values = files.get(section)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                paths.append(recording_dir / value)
    return paths


def _analysis_zarr_path(recording_dir: Path) -> Path:
    return recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"


def _unique_target_path(processed_root: Path, batch_path: Path) -> Path:
    base = processed_root / batch_path.name
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = processed_root / f"{batch_path.name}.{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def build_finalize_plan(
    batch_path: Path,
    *,
    recordings_root: Path,
    organize_logs: Iterable[Path] = (),
    import_status: Optional[ImportLogStatus] = None,
    processed_root: Optional[Path] = None,
    require_analysis_zarr: bool = True,
    allow_preflight_failures: bool = False,
    allow_organize_warnings: bool = False,
    delete: bool = False,
) -> FinalizePlan:
    batch = batch_path.expanduser()
    blockers: list[str] = []
    warnings: list[str] = []
    if not batch.exists():
        blockers.append(f"staging batch does not exist: {batch}")
    elif not batch.is_dir():
        blockers.append(f"staging batch is not a directory: {batch}")

    manifest_recordings = _recordings_from_manifests(batch, recordings_root)
    log_recordings, organize_warnings = _recordings_from_organize_logs(batch, organize_logs)
    recordings = sorted({*manifest_recordings, *log_recordings})
    if organize_warnings:
        warnings.extend(organize_warnings)
        if not allow_organize_warnings:
            blockers.extend([f"organize warning: {message}" for message in organize_warnings])
    if not recordings:
        blockers.append("no organized recording manifests or organize-log entries matched this staging batch")

    manifest_paths: list[Path] = []
    missing_files: list[str] = []
    missing_zarrs: list[Path] = []
    for recording_dir in recordings:
        manifest_path = recording_dir / "recording_manifest.json"
        manifest_paths.append(manifest_path)
        manifest = _load_json(manifest_path)
        if manifest is None:
            blockers.append(f"missing or invalid recording manifest: {manifest_path}")
            continue
        if not _manifest_source_matches_batch(manifest, batch):
            blockers.append(f"manifest source_dir does not point into batch: {manifest_path}")
        preflight = manifest.get("preflight")
        if (
            isinstance(preflight, Mapping)
            and str(preflight.get("status", "")).lower() == "fail"
            and not allow_preflight_failures
        ):
            blockers.append(f"preflight failed in manifest: {manifest_path}")
        file_paths = _manifest_file_paths(recording_dir, manifest)
        if not file_paths:
            warnings.append(f"manifest has no files.raw/cams/derived entries: {manifest_path}")
        for file_path in file_paths:
            if not file_path.exists():
                missing_files.append(str(file_path))
        zarr_path = _analysis_zarr_path(recording_dir)
        if require_analysis_zarr and not zarr_path.exists():
            missing_zarrs.append(zarr_path)

        if import_status is not None:
            key = str(_safe_resolve(recording_dir))
            if key in import_status.failed:
                blockers.append(f"import failed for {recording_dir}: {import_status.failed[key]}")
            elif key in import_status.skipped:
                blockers.append(f"import skipped for {recording_dir}: {import_status.skipped[key]}")

    if missing_files:
        blockers.extend([f"manifest-listed file missing: {path}" for path in missing_files])
    if missing_zarrs:
        blockers.extend([f"analysis zarr missing: {path}" for path in missing_zarrs])

    action = "delete" if delete else "move"
    target_path = None
    if not delete:
        root = processed_root or (batch.parent / ".processed")
        target_path = _unique_target_path(root, batch)
        if _safe_resolve(root) == _safe_resolve(batch) or _path_is_relative_to(root, batch):
            blockers.append(f"processed root must not be inside the staging batch: {root}")

    return FinalizePlan(
        batch_path=batch,
        recordings=recordings,
        status="ready" if not blockers else "blocked",
        action=action,
        target_path=target_path,
        blockers=blockers,
        warnings=warnings,
        manifest_paths=manifest_paths,
        missing_files=missing_files,
        missing_zarrs=missing_zarrs,
    )


def apply_finalize_plan(plan: FinalizePlan) -> FinalizePlan:
    if plan.status != "ready":
        raise RuntimeError(f"Refusing to finalize blocked batch: {plan.batch_path}")
    if plan.action == "delete":
        shutil.rmtree(plan.batch_path)
    else:
        if plan.target_path is None:
            raise RuntimeError("move finalization requires target_path")
        plan.target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(plan.batch_path), str(plan.target_path))
    plan.applied = True
    return plan


def _discover_batches(paths: list[Path], *, process_all: bool) -> list[Path]:
    if not paths:
        paths = [_resolve_default_staging_root()]
    batches: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if process_all:
            batches.extend(
                child
                for child in sorted(expanded.iterdir())
                if child.is_dir() and not child.name.startswith(".")
            )
        else:
            batches.append(expanded)
    return batches


def _print_plan(plan: FinalizePlan) -> None:
    print(f"Batch: {plan.batch_path}")
    print(f"  status: {plan.status}")
    print(f"  action: {plan.action}")
    if plan.target_path is not None:
        print(f"  target: {plan.target_path}")
    print(f"  recordings: {len(plan.recordings)}")
    for recording in plan.recordings:
        print(f"    - {recording}")
    if plan.warnings:
        print("  warnings:")
        for warning in plan.warnings:
            print(f"    - {warning}")
    if plan.blockers:
        print("  blockers:")
        for blocker in plan.blockers:
            print(f"    - {blocker}")
    print("")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify organized/imported recordings, then move/delete completed staging batches.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Staging batch directories, or staging roots when --process-all is used.",
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=_resolve_default_recordings_root(),
        help="Root containing organized recording directories.",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Treat each immediate child directory of each path as a staging batch.",
    )
    parser.add_argument(
        "--organize-log",
        type=Path,
        action="append",
        default=[],
        help="Optional organize_recordings JSONL log. May be repeated.",
    )
    parser.add_argument(
        "--import-log",
        type=Path,
        action="append",
        default=[],
        help="Optional import_organized_recordings_analysis JSONL log. Failed/skipped rows block finalization.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        help="Directory for moved finalized batches (default: <staging-root>/.processed).",
    )
    parser.add_argument(
        "--allow-missing-analysis-zarr",
        action="store_true",
        help="Allow finalization after organize only. Default requires each recording analysis zarr to exist.",
    )
    parser.add_argument(
        "--allow-preflight-failures",
        action="store_true",
        help="Do not block on manifest preflight.status=fail.",
    )
    parser.add_argument(
        "--allow-organize-warnings",
        action="store_true",
        help="Do not block on warnings found in supplied organize logs.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete verified staging batches instead of moving them to .processed.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply finalization. Default is dry-run.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Optional JSONL log path for this finalization run.",
    )
    args = parser.parse_args(argv)

    batches = _discover_batches(args.paths, process_all=bool(args.process_all))
    import_status = read_import_log_status(args.import_log) if args.import_log else None
    logger: Optional[JsonLogger] = None
    if args.log_path is not None:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = JsonLogger(args.log_path, make_run_id())

    plans = [
        build_finalize_plan(
            batch,
            recordings_root=args.recordings_root.expanduser(),
            organize_logs=args.organize_log,
            import_status=import_status,
            processed_root=args.processed_root.expanduser() if args.processed_root else None,
            require_analysis_zarr=not bool(args.allow_missing_analysis_zarr),
            allow_preflight_failures=bool(args.allow_preflight_failures),
            allow_organize_warnings=bool(args.allow_organize_warnings),
            delete=bool(args.delete),
        )
        for batch in batches
    ]

    exit_code = 0
    for plan in plans:
        if args.apply and plan.status == "ready":
            apply_finalize_plan(plan)
        elif args.apply and plan.status != "ready":
            exit_code = 1
        if plan.status != "ready":
            exit_code = 1
        if logger is not None:
            logger.log("finalize_plan", **plan.to_dict())

    if logger is not None:
        logger.log(
            "run_end",
            ready=sum(1 for plan in plans if plan.status == "ready"),
            blocked=sum(1 for plan in plans if plan.status != "ready"),
            applied=sum(1 for plan in plans if plan.applied),
        )
        logger.close()

    if args.json:
        print(json.dumps({"plans": [plan.to_dict() for plan in plans]}, indent=2))
    else:
        for plan in plans:
            _print_plan(plan)
        ready = sum(1 for plan in plans if plan.status == "ready")
        blocked = len(plans) - ready
        applied = sum(1 for plan in plans if plan.applied)
        print(f"Summary: ready={ready} blocked={blocked} applied={applied}")
        if not args.apply:
            print("Dry-run only; pass --apply to finalize ready batches.")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
