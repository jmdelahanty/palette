#!/usr/bin/env python3
"""Batch import analysis Zarr archives from already organized recordings.

This is intentionally import-only. It creates/updates analysis archives with
recording metadata, source-video metadata, and optional H5 stimulus metadata.
It does not run detect, refine, or keypoints. When --registry is provided,
current-profile imports are receipt-finalized and already-bound existing
archives are verified before their non-identity registry metadata is refreshed.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import zarr

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    load_acquisition_authority_publication_status,
)
from fisheye.registry.shadow_publish import shadow_synchronize_recording_import
from fisheye.shared.batch_logging import JsonLogger, make_run_id, utc_now
from fisheye.shared.recording_import_receipt import RecordingImportReceipt
from fisheye.utils.import_recording_analysis import (
    RecordingAnalysisPlan,
    RecordingImportOptions,
    process_recording_import,
    resolve_single_recording_plan,
    stimulus_runs_present,
)
from fisheye.shared.recording_preflight import preflight_gate_reason


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


@dataclass(frozen=True)
class OrganizedImportPlan:
    recording_dir: Path
    h5_path: Optional[Path]
    cam_video: Optional[Path]
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    stimulus_present: Optional[bool] = None


@dataclass(frozen=True)
class RegistrySyncResult:
    ok: bool
    registry_path: Path
    zarr_path: Path
    dataset_id: Optional[str] = None
    error: Optional[str] = None


def _resolve_root(arg_root: Optional[Path], *, organize_logs: list[Path]) -> Path:
    if arg_root is not None:
        return arg_root.expanduser().resolve()
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if organize_logs:
        return Path.cwd()
    return DEFAULT_RECORDINGS_ROOT


def _resolve_log_dir(arg_log_dir: Optional[Path], root: Path) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir.expanduser().resolve()
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve() / "import_organized_recordings_analysis"
    return root / "logs" / "import_organized_recordings_analysis"


def _find_h5_recording_dirs(root: Path, *, recursive: bool) -> set[Path]:
    pattern: Iterable[Path]
    if recursive:
        pattern = root.rglob("raw/*.h5")
    else:
        pattern = root.glob("*/raw/*.h5")
    return {path.parent.parent for path in pattern if path.parent.name == "raw"}


def _find_video_recording_dirs(root: Path, *, recursive: bool) -> set[Path]:
    recording_dirs: set[Path] = set()
    if (root / "cams").is_dir():
        recording_dirs.add(root)
    pattern: Iterable[Path]
    if recursive:
        pattern = root.rglob("cams/*.mp4")
    else:
        pattern = root.glob("*/cams/*.mp4")
    for path in pattern:
        if path.parent.name == "cams":
            recording_dirs.add(path.parent.parent)
    return recording_dirs


def _read_organize_log_recording_dirs(log_paths: Iterable[Path]) -> set[Path]:
    recording_dirs: set[Path] = set()
    for log_path in log_paths:
        path = log_path.expanduser()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") != "recording_applied":
                    continue
                dest_dir = payload.get("dest_dir")
                if isinstance(dest_dir, str) and dest_dir.strip():
                    recording_dirs.add(Path(dest_dir).expanduser().resolve())
    return recording_dirs


def discover_recording_dirs(
    root: Path,
    *,
    recursive: bool,
    import_stimulus: bool,
    organize_logs: Iterable[Path] = (),
) -> list[Path]:
    from_logs = _read_organize_log_recording_dirs(organize_logs)
    if from_logs:
        return sorted(from_logs)
    if import_stimulus:
        return sorted(_find_h5_recording_dirs(root, recursive=recursive))
    return sorted(_find_video_recording_dirs(root, recursive=recursive))


def _existing_analysis_complete(
    zarr_path: Path,
    *,
    require_stimulus: bool,
) -> tuple[bool, str]:
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        publication = load_acquisition_authority_publication_status(root)
    except Exception as exc:
        return False, f"acquisition authority is incomplete: {exc}"
    if publication.status != ACQUISITION_AUTHORITY_PUBLISHED:
        return False, f"acquisition authority status is {publication.status!r}"
    if require_stimulus and not stimulus_runs_present(zarr_path):
        return False, "stimulus runs are missing"
    return True, "analysis import completion contract is satisfied"


def build_plans(
    recording_dirs: Iterable[Path],
    *,
    import_stimulus: bool,
    skip_existing: bool,
    allow_preflight_failures: bool,
    check_stimulus: bool,
) -> list[OrganizedImportPlan]:
    plans: list[OrganizedImportPlan] = []
    for recording_dir in sorted({Path(path).expanduser().resolve() for path in recording_dirs}):
        try:
            plan = resolve_single_recording_plan(
                recording_dir=recording_dir,
                require_h5=bool(import_stimulus),
            )
        except ValueError as exc:
            zarr_path = recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"
            plans.append(
                OrganizedImportPlan(
                    recording_dir=recording_dir,
                    h5_path=None,
                    cam_video=None,
                    zarr_path=zarr_path,
                    status="missing",
                    reason=str(exc),
                )
            )
            continue

        status = "ok"
        reason: Optional[str] = preflight_gate_reason(
            recording_dir,
            allow_failures=bool(allow_preflight_failures),
        )
        if reason is not None:
            status = "missing"
        elif skip_existing and plan.zarr_path.exists():
            complete, completion_reason = _existing_analysis_complete(
                plan.zarr_path,
                require_stimulus=bool(import_stimulus),
            )
            if complete:
                status = "skipped"
                reason = completion_reason
            else:
                reason = f"resuming incomplete analysis zarr: {completion_reason}"

        stimulus_present: Optional[bool] = None
        if check_stimulus and plan.zarr_path.exists():
            stimulus_present = stimulus_runs_present(plan.zarr_path)

        plans.append(
            OrganizedImportPlan(
                recording_dir=plan.recording_dir,
                h5_path=plan.h5_path,
                cam_video=plan.cam_video,
                zarr_path=plan.zarr_path,
                status=status,
                reason=reason,
                stimulus_present=stimulus_present,
            )
        )
    return plans


def _print_plans(plans: list[OrganizedImportPlan]) -> None:
    if not plans:
        print("No organized recordings found.")
        return
    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
        print(f"Recording: {plan.recording_dir.name}")
        print(f"  recording_dir: {plan.recording_dir}")
        print(f"  h5: {plan.h5_path if plan.h5_path is not None else 'none'}")
        print(f"  cam: {plan.cam_video if plan.cam_video is not None else 'MISSING'}")
        print(f"  analysis_zarr: {plan.zarr_path}")
        if plan.stimulus_present is not None:
            label = "present" if plan.stimulus_present else "missing"
            print(f"  stimulus_runs: {label}")
        print(f"  status: {plan.status}" + (f" ({plan.reason})" if plan.reason else ""))
        print("")
    print("Summary:")
    for key in ("ok", "skipped", "missing"):
        print(f"  {key}: {counts.get(key, 0)}")


def _make_import_options(args: argparse.Namespace) -> RecordingImportOptions:
    return RecordingImportOptions(
        import_video_metadata=bool(args.import_video_metadata),
        video_metadata_overwrite=bool(args.video_metadata_overwrite),
        import_stimulus=bool(args.import_stimulus),
        stimulus_always=bool(args.stimulus_always),
        stimulus_run_name=args.stimulus_run_name,
        stimulus_overwrite=bool(args.stimulus_overwrite),
        stimulus_quiet=bool(args.stimulus_quiet),
        allow_preflight_failures=bool(args.allow_preflight_failures),
        stimulus_metadata_and_calibration_only=bool(
            args.stimulus_metadata_and_calibration_only
        ),
    )


def _log_plan(logger: Optional[JsonLogger], plan: OrganizedImportPlan) -> None:
    if logger is None:
        return
    logger.log(
        "recording_plan",
        recording_dir=str(plan.recording_dir),
        h5_path=str(plan.h5_path) if plan.h5_path is not None else None,
        cam_video=str(plan.cam_video) if plan.cam_video is not None else None,
        zarr_path=str(plan.zarr_path),
        status=plan.status,
        reason=plan.reason,
        stimulus_present=plan.stimulus_present,
    )


def _sync_registry(
    registry_path: Path,
    zarr_path: Path,
    *,
    receipt: RecordingImportReceipt | None = None,
) -> RegistrySyncResult:
    resolved_registry = registry_path.expanduser().resolve()
    try:
        publication = shadow_synchronize_recording_import(
            canonical_registry=resolved_registry,
            zarr_path=zarr_path,
            receipt=receipt,
            decided_by="fisheye.utils.import_organized_recordings_analysis",
        )
        dataset_id = str(publication.mutation_result["dataset_id"])
    except Exception as exc:
        return RegistrySyncResult(
            ok=False,
            registry_path=resolved_registry,
            zarr_path=zarr_path,
            error=str(exc),
        )
    return RegistrySyncResult(
        ok=True,
        registry_path=resolved_registry,
        zarr_path=zarr_path,
        dataset_id=dataset_id,
    )


def _log_registry_sync(
    logger: Optional[JsonLogger],
    *,
    event: str,
    recording_dir: Path,
    result: RegistrySyncResult,
) -> None:
    if logger is None:
        return
    logger.log(
        event,
        recording_dir=str(recording_dir),
        zarr_path=str(result.zarr_path),
        registry_path=str(result.registry_path),
        dataset_id=result.dataset_id,
        error=result.error,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create/import analysis zarrs for organized recordings only. "
            "This command never runs detect/refine/keypoints."
        ),
    )
    parser.add_argument(
        "recordings_root",
        nargs="?",
        type=Path,
        help="Root recordings directory (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for organized recordings.")
    parser.add_argument(
        "--organize-log",
        type=Path,
        action="append",
        default=[],
        help="Read recording_applied destinations from an organize_recordings JSONL log. May be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned imports without writing zarrs.")
    parser.add_argument("--apply", action="store_true", help="Create/update analysis zarrs.")
    parser.add_argument("--overwrite", action="store_true", help="Process recordings even when analysis zarr already exists.")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Alias for --overwrite; retained for parity with older batch tools.",
    )
    parser.add_argument(
        "--import-video-metadata",
        dest="import_video_metadata",
        action="store_true",
        help="Import source-video metadata into root/raw_video attrs (default).",
    )
    parser.add_argument(
        "--no-import-video-metadata",
        dest="import_video_metadata",
        action="store_false",
        help="Skip source-video metadata import.",
    )
    parser.add_argument(
        "--video-metadata-overwrite",
        action="store_true",
        help="Overwrite existing source-video metadata attrs when importing.",
    )
    parser.add_argument(
        "--import-stimulus",
        dest="import_stimulus",
        action="store_true",
        help="Import H5 stimulus metadata into analysis/stimulus_runs (default).",
    )
    parser.add_argument(
        "--no-import-stimulus",
        dest="import_stimulus",
        action="store_false",
        help="Skip stimulus import.",
    )
    parser.add_argument(
        "--recording-only",
        action="store_true",
        help="Process camera-video-only recordings without requiring raw/*.h5.",
    )
    parser.set_defaults(import_stimulus=True, import_video_metadata=True)
    parser.add_argument("--stimulus-always", action="store_true", help="Run stimulus import even if runs exist.")
    parser.add_argument("--stimulus-run-name", type=str, help="Optional stimulus run name.")
    parser.add_argument("--stimulus-overwrite", action="store_true", help="Overwrite existing stimulus run name.")
    parser.add_argument("--stimulus-quiet", action="store_true", help="Suppress verbose stimulus import output.")
    parser.add_argument(
        "--stimulus-metadata-and-calibration-only",
        action="store_true",
        help=(
            "Import stimulus events, protocol metadata, and selected calibration "
            "while omitting H5 coordinate surfaces without canonical array-level "
            "identity."
        ),
    )
    parser.add_argument(
        "--allow-preflight-failures",
        action="store_true",
        help="Proceed even if recording_manifest.json marks preflight.status=fail.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/import_organized_recordings_analysis or <root>/logs/import_organized_recordings_analysis).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help=(
            "Registry SQLite path to finalize after successful imports. "
            "Existing current-profile archives require a prior receipt binding; "
            "unprofiled compatibility archives retain generic scanning."
        ),
    )
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging.")

    args = parser.parse_args(argv)
    if args.recording_only:
        args.import_stimulus = False
    if args.no_skip_existing:
        args.overwrite = True
    if not args.apply:
        args.dry_run = True

    root = _resolve_root(args.recordings_root, organize_logs=list(args.organize_log))
    if not args.organize_log and not root.exists():
        print(f"Recordings root not found: {root}")
        return 1

    run_id = make_run_id()
    logger: Optional[JsonLogger] = None
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, root)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"import_organized_recordings_analysis_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            recordings_root=str(root),
            recursive=bool(args.recursive),
            organize_logs=[str(path) for path in args.organize_log],
            dry_run=bool(args.dry_run),
            apply=bool(args.apply),
            skip_existing=not bool(args.overwrite),
            import_video_metadata=bool(args.import_video_metadata),
            import_stimulus=bool(args.import_stimulus),
            stimulus_metadata_and_calibration_only=bool(
                args.stimulus_metadata_and_calibration_only
            ),
            allow_preflight_failures=bool(args.allow_preflight_failures),
            registry=str(args.registry.expanduser()) if args.registry is not None else None,
            created_at_utc=utc_now(),
        )

    recording_dirs = discover_recording_dirs(
        root,
        recursive=bool(args.recursive),
        import_stimulus=bool(args.import_stimulus),
        organize_logs=args.organize_log,
    )
    plans = build_plans(
        recording_dirs,
        import_stimulus=bool(args.import_stimulus),
        skip_existing=not bool(args.overwrite),
        allow_preflight_failures=bool(args.allow_preflight_failures),
        check_stimulus=bool(args.import_stimulus),
    )

    if args.dry_run:
        print("Planned organized recording analysis imports (dry-run):")
        _print_plans(plans)
        for plan in plans:
            _log_plan(logger, plan)
        if logger is not None:
            logger.log("run_end", ok=0, failed=0, skipped=0, missing=0, dry_run=True)
            logger.close()
        return 0

    opts = _make_import_options(args)
    ok = 0
    failed = 0
    skipped = 0
    missing = 0
    registry_synced = 0
    registry_failed = 0
    for plan in plans:
        _log_plan(logger, plan)
        if plan.status == "missing":
            missing += 1
            print(f"Skipping missing recording: {plan.recording_dir} ({plan.reason})")
            if logger is not None:
                logger.log(
                    "recording_skipped",
                    recording_dir=str(plan.recording_dir),
                    zarr_path=str(plan.zarr_path),
                    status="missing",
                    reason=plan.reason,
                )
            continue
        if plan.status == "skipped":
            if args.registry is not None:
                sync_result = _sync_registry(args.registry, plan.zarr_path)
                if sync_result.ok:
                    registry_synced += 1
                    print(
                        "Registry synced existing analysis zarr: "
                        f"{plan.zarr_path} ({sync_result.dataset_id})"
                    )
                    _log_registry_sync(
                        logger,
                        event="registry_sync_ok",
                        recording_dir=plan.recording_dir,
                        result=sync_result,
                    )
                else:
                    registry_failed += 1
                    failed += 1
                    print(
                        "Registry sync failed for existing analysis zarr: "
                        f"{plan.zarr_path} error={sync_result.error}"
                    )
                    _log_registry_sync(
                        logger,
                        event="registry_sync_failed",
                        recording_dir=plan.recording_dir,
                        result=sync_result,
                    )
                    continue
            skipped += 1
            print(f"Skipping existing analysis zarr: {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "recording_skipped",
                    recording_dir=str(plan.recording_dir),
                    zarr_path=str(plan.zarr_path),
                    status="skipped",
                    reason=plan.reason,
                )
            continue

        assert plan.cam_video is not None
        import_plan = RecordingAnalysisPlan(
            recording_dir=plan.recording_dir,
            h5_path=plan.h5_path,
            cam_video=plan.cam_video,
            zarr_path=plan.zarr_path,
        )
        if logger is not None:
            logger.log(
                "recording_start",
                recording_dir=str(plan.recording_dir),
                h5_path=str(plan.h5_path) if plan.h5_path is not None else None,
                cam_video=str(plan.cam_video),
                zarr_path=str(plan.zarr_path),
            )
        result = process_recording_import(import_plan, opts, logger=(logger.log if logger else None))
        if not result.ok:
            failed += 1
            print(
                f"Import failed for {plan.recording_dir}: step={result.failed_step or 'unknown'}"
                + (f" returncode={result.returncode}" if result.returncode is not None else "")
                + (f" error={result.error}" if result.error else "")
            )
            if logger is not None:
                logger.log(
                    "recording_failed",
                    recording_dir=str(plan.recording_dir),
                    zarr_path=str(plan.zarr_path),
                    step=result.failed_step,
                    returncode=result.returncode,
                    error=result.error,
                )
            continue
        if args.registry is not None:
            sync_result = _sync_registry(
                args.registry,
                plan.zarr_path,
                receipt=result.receipt,
            )
            if not sync_result.ok:
                registry_failed += 1
                failed += 1
                print(
                    "Import succeeded but registry sync failed for "
                    f"{plan.recording_dir}: error={sync_result.error}"
                )
                _log_registry_sync(
                    logger,
                    event="registry_sync_failed",
                    recording_dir=plan.recording_dir,
                    result=sync_result,
                )
                if logger is not None:
                    logger.log(
                        "recording_failed",
                        recording_dir=str(plan.recording_dir),
                        zarr_path=str(plan.zarr_path),
                        step="registry_sync",
                        returncode=None,
                        error=sync_result.error,
                    )
                continue
            registry_synced += 1
            print(f"Registry synced analysis zarr: {plan.zarr_path} ({sync_result.dataset_id})")
            _log_registry_sync(
                logger,
                event="registry_sync_ok",
                recording_dir=plan.recording_dir,
                result=sync_result,
            )

        ok += 1
        print(f"Imported analysis zarr: {plan.zarr_path}")
        if logger is not None:
            logger.log(
                "recording_ok",
                recording_dir=str(plan.recording_dir),
                zarr_path=str(plan.zarr_path),
            )

    print("Summary:")
    print(f"  ok: {ok}")
    print(f"  failed: {failed}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    if args.registry is not None:
        print(f"  registry_synced: {registry_synced}")
        print(f"  registry_failed: {registry_failed}")
    if logger is not None:
        logger.log(
            "run_end",
            ok=ok,
            failed=failed,
            skipped=skipped,
            missing=missing,
            registry_synced=registry_synced,
            registry_failed=registry_failed,
            dry_run=False,
        )
        logger.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
