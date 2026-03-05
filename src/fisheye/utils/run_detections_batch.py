#!/usr/bin/env python3
"""Batch-run registry-backed detection on analysis zarr archives."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.environment import resolve_log_dir
from fisheye.shared.environment import resolve_recording_roots
from fisheye.shared.zarr_discovery import discover_registry_zarrs
from fisheye.utils.run_detect_with_registry_model import run_detect_with_registry_model

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore


STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_MISSING = "missing"
STATUS_FAILED = "failed"

REASON_ANALYSIS_ZARR_MISSING = "analysis_zarr_missing"
REASON_ANALYSIS_ZARR_OPEN_FAILED = "analysis_zarr_open_failed"
REASON_NOT_ANALYSIS_ZARR = "not_analysis_zarr"
REASON_CAMERA_VIDEO_MISSING = "camera_video_missing"
REASON_CAMERA_VIDEO_AMBIGUOUS = "camera_video_ambiguous"
REASON_CAMS_DIR_MISSING = "cams_directory_missing"
REASON_BACKGROUND_MISSING = "background_missing"
REASON_DETECTION_TUNING_MISSING = "detection_tuning_missing"
REASON_DETECT_ALREADY_PRESENT = "detect_already_present"
REASON_REGISTRY_MISSING = "registry_missing"


@dataclass
class DetectPlan:
    zarr_path: Path
    recording_dir: Path
    video_path: Optional[Path]
    status: str
    reason: Optional[str] = None
    detect_present: bool = False
    background_present: bool = False
    tuning_present: bool = False


class JsonLogger(SharedJsonLogger):
    def log(self, event: str, *, zarr: str, status: str, reason: Optional[str] = None, **fields: object) -> None:
        payload: dict[str, object] = {"zarr": zarr, "status": status}
        if reason is not None:
            payload["reason"] = reason
        payload.update(fields)
        super().log(event, **payload)


def _load_paths_file(path: Path) -> List[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _resolve_input_paths(paths: Sequence[Path], file_lists: Sequence[Path]) -> List[Path]:
    input_paths: List[Path] = []
    input_paths.extend(paths)
    for file_path in file_lists:
        input_paths.extend(_load_paths_file(file_path))
    return resolve_recording_roots(input_paths)


def _resolve_log_dir(arg_log_dir: Optional[Path], inputs: Sequence[Path]) -> Path:
    log_roots: list[Path] = []
    for path in inputs:
        if path.suffix == ".zarr":
            log_roots.append(_infer_recording_dir(path))
        else:
            log_roots.append(path)
    return resolve_log_dir(arg_log_dir, log_roots, log_subdir="run_detections_batch")


def _run_id() -> str:
    return make_run_id()


def _is_analysis_zarr(path: Path) -> bool:
    return path.name.endswith("_analysis.zarr")


def _candidate_paths(path: Path, recursive: bool) -> Iterable[Path]:
    if path.suffix == ".zarr":
        yield path
        return

    expanded = path.expanduser()
    if not expanded.exists() or not expanded.is_dir():
        return

    if recursive:
        yield from expanded.rglob("*_analysis.zarr")
        return

    yield from expanded.glob("*_analysis.zarr")
    yield from expanded.glob("zarr/*_analysis.zarr")
    yield from expanded.glob("*/zarr/*_analysis.zarr")


def _discover_analysis_zarrs(paths: Sequence[Path], recursive: bool) -> List[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        candidate = raw_path.expanduser()
        if candidate.suffix == ".zarr" and not candidate.exists():
            resolved = candidate.resolve()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                ordered.append(resolved)
            continue

        for discovered in _candidate_paths(candidate, recursive):
            resolved = discovered.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(resolved)
    ordered.sort(key=lambda item: str(item))
    return ordered


def _discover_analysis_zarrs_from_registry(
    *,
    registry_path: Path,
    scope_paths: Sequence[Path],
    rig_id: Optional[str] = None,
    arena_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    path_contains: Optional[str] = None,
    skip_existing: bool = False,
) -> List[Path]:
    """Query the registry for analysis zarr paths, optionally scoped to directories.

    When *skip_existing* is True, recordings whose ``detect`` step status is
    already ``'ok'`` in the registry are excluded at the SQL level, avoiding
    unnecessary filesystem I/O during plan building.
    """
    return discover_registry_zarrs(
        registry_path=registry_path,
        scope_paths=scope_paths,
        zarr_use="analysis",
        rig_id=rig_id,
        arena_id=arena_id,
        camera_id=camera_id,
        path_contains=path_contains,
        exclude_step_ok="detect" if skip_existing else None,
        zarr_suffix="_analysis.zarr",
    )


def _infer_recording_dir(zarr_path: Path) -> Path:
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    return zarr_path.parent


def _resolve_video_path(recording_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    cams_dir = recording_dir / "cams"
    if not cams_dir.exists() or not cams_dir.is_dir():
        return None, REASON_CAMS_DIR_MISSING
    mp4s = sorted(cams_dir.glob("*.mp4"))
    if not mp4s:
        return None, REASON_CAMERA_VIDEO_MISSING
    if len(mp4s) > 1:
        return None, REASON_CAMERA_VIDEO_AMBIGUOUS
    return mp4s[0].resolve(), None


def _read_group_attrs(zarr_path: Path, group_name: str) -> Optional[dict[str, object]]:
    group_dir = zarr_path / group_name
    zarr_json = group_dir / "zarr.json"
    if zarr_json.exists():
        try:
            payload = json.loads(zarr_json.read_text(encoding="utf-8"))
        except Exception:
            return None
        attrs = payload.get("attributes")
        return attrs if isinstance(attrs, dict) else {}

    zattrs = group_dir / ".zattrs"
    if zattrs.exists():
        try:
            payload = json.loads(zattrs.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else {}

    return None


def _analysis_metadata_readable(zarr_path: Path) -> bool:
    root_zarr_json = zarr_path / "zarr.json"
    if root_zarr_json.exists():
        try:
            json.loads(root_zarr_json.read_text(encoding="utf-8"))
            return True
        except Exception:
            return False
    if (zarr_path / ".zgroup").exists():
        return True
    return False


def _has_group_latest(zarr_path: Path, group_name: str) -> bool:
    attrs = _read_group_attrs(zarr_path, group_name)
    if not attrs:
        return False
    return bool(attrs.get("latest"))


def _has_detection_tuning(zarr_path: Path) -> bool:
    attrs = _read_group_attrs(zarr_path, "analysis_metadata")
    if not attrs:
        return False
    return "detection_tuning" in attrs


def _build_plan_for_zarr(
    *,
    zarr_path: Path,
    skip_existing: bool,
    require_background: bool,
    require_tuning: bool,
) -> DetectPlan:
    recording_dir = _infer_recording_dir(zarr_path)
    if not _is_analysis_zarr(zarr_path):
        return DetectPlan(
            zarr_path=zarr_path,
            recording_dir=recording_dir,
            video_path=None,
            status=STATUS_MISSING,
            reason=REASON_NOT_ANALYSIS_ZARR,
        )

    if not zarr_path.exists():
        return DetectPlan(
            zarr_path=zarr_path,
            recording_dir=recording_dir,
            video_path=None,
            status=STATUS_MISSING,
            reason=REASON_ANALYSIS_ZARR_MISSING,
        )

    if not _analysis_metadata_readable(zarr_path):
        return DetectPlan(
            zarr_path=zarr_path,
            recording_dir=recording_dir,
            video_path=None,
            status=STATUS_MISSING,
            reason=REASON_ANALYSIS_ZARR_OPEN_FAILED,
        )

    detect_present = _has_group_latest(zarr_path, "detect_runs")
    background_present = _has_group_latest(zarr_path, "background_runs")
    tuning_present = _has_detection_tuning(zarr_path)

    if require_background and not background_present:
        return DetectPlan(
            zarr_path=zarr_path,
            recording_dir=recording_dir,
            video_path=None,
            status=STATUS_MISSING,
            reason=REASON_BACKGROUND_MISSING,
            detect_present=detect_present,
            background_present=background_present,
            tuning_present=tuning_present,
        )

    if require_tuning and not tuning_present:
        return DetectPlan(
            zarr_path=zarr_path,
            recording_dir=recording_dir,
            video_path=None,
            status=STATUS_MISSING,
            reason=REASON_DETECTION_TUNING_MISSING,
            detect_present=detect_present,
            background_present=background_present,
            tuning_present=tuning_present,
        )

    video_path, video_reason = _resolve_video_path(recording_dir)
    if video_path is None:
        return DetectPlan(
            zarr_path=zarr_path,
            recording_dir=recording_dir,
            video_path=None,
            status=STATUS_MISSING,
            reason=video_reason,
            detect_present=detect_present,
            background_present=background_present,
            tuning_present=tuning_present,
        )

    if skip_existing and detect_present:
        return DetectPlan(
            zarr_path=zarr_path,
            recording_dir=recording_dir,
            video_path=video_path,
            status=STATUS_SKIPPED,
            reason=REASON_DETECT_ALREADY_PRESENT,
            detect_present=detect_present,
            background_present=background_present,
            tuning_present=tuning_present,
        )

    return DetectPlan(
        zarr_path=zarr_path,
        recording_dir=recording_dir,
        video_path=video_path,
        status=STATUS_OK,
        detect_present=detect_present,
        background_present=background_present,
        tuning_present=tuning_present,
    )


def _build_plans(
    zarr_paths: Sequence[Path],
    *,
    skip_existing: bool,
    require_background: bool,
    require_tuning: bool,
) -> List[DetectPlan]:
    plans: List[DetectPlan] = []
    for zarr_path in sorted(zarr_paths, key=lambda item: str(item)):
        plans.append(
            _build_plan_for_zarr(
                zarr_path=zarr_path,
                skip_existing=skip_existing,
                require_background=require_background,
                require_tuning=require_tuning,
            )
        )
    return plans


def _apply_registry_prereq(plans: Sequence[DetectPlan], *, registry_path: Path) -> List[DetectPlan]:
    if registry_path.exists():
        return list(plans)

    updated: List[DetectPlan] = []
    for plan in plans:
        if plan.status != STATUS_OK:
            updated.append(plan)
            continue
        updated.append(
            DetectPlan(
                zarr_path=plan.zarr_path,
                recording_dir=plan.recording_dir,
                video_path=plan.video_path,
                status=STATUS_MISSING,
                reason=REASON_REGISTRY_MISSING,
                detect_present=plan.detect_present,
                background_present=plan.background_present,
                tuning_present=plan.tuning_present,
            )
        )
    return updated


def _counts_from_plans(plans: Sequence[DetectPlan]) -> dict[str, int]:
    counts = {STATUS_OK: 0, STATUS_SKIPPED: 0, STATUS_MISSING: 0}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
    return counts


def _plan_payload(plan: DetectPlan) -> dict[str, object]:
    return {
        "recording": str(plan.recording_dir),
        "zarr": str(plan.zarr_path),
        "video": str(plan.video_path) if plan.video_path else None,
        "status": plan.status,
        "reason": plan.reason,
        "detect_present": plan.detect_present,
        "background_present": plan.background_present,
        "tuning_present": plan.tuning_present,
    }


def _print_plan(plans: Sequence[DetectPlan]) -> None:
    counts = _counts_from_plans(plans)
    for plan in plans:
        print(f"Recording: {plan.recording_dir}")
        print(f"  zarr: {plan.zarr_path}")
        print(f"  video: {plan.video_path or 'MISSING'}")
        print(f"  status: {plan.status}")
        if plan.reason:
            print(f"  reason: {plan.reason}")
    print("\nSummary:")
    print(f"  ok: {counts.get(STATUS_OK, 0)}")
    print(f"  skipped: {counts.get(STATUS_SKIPPED, 0)}")
    print(f"  missing: {counts.get(STATUS_MISSING, 0)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch run registry-backed detect on analysis zarr archives.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots, recording directories, or analysis zarr paths.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one root/recording/analysis-zarr path per line.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan roots for *_analysis.zarr.")
    parser.add_argument(
        "--source",
        choices=["filesystem", "registry"],
        default="filesystem",
        help="Discovery source for analysis zarrs (default: filesystem).",
    )
    parser.add_argument(
        "--emit-paths",
        action="store_true",
        help="Print discovered zarr paths (one per line) and exit.",
    )

    apply_group = parser.add_mutually_exclusive_group()
    apply_group.add_argument("--apply", action="store_true", help="Run detect for planned analysis zarrs.")
    apply_group.add_argument("--dry-run", action="store_true", help="Show planned detections without running.")

    parser.add_argument("--overwrite", action="store_true", help="Run detection even when detect runs already exist.")
    parser.add_argument("--require-background", action="store_true", help="Require background_runs/latest before detect.")
    parser.add_argument(
        "--no-require-background",
        action="store_true",
        help="Allow detect planning when background_runs/latest is missing.",
    )
    parser.add_argument(
        "--require-tuning",
        action="store_true",
        help="Require analysis_metadata attrs to include detection_tuning.",
    )

    parser.add_argument("--registry", type=Path, help="Optional registry sqlite path.")
    parser.add_argument("--set-id", type=str, help="Optional detect set filter during model resolution.")
    parser.add_argument("--require-unique", action="store_true", help="Fail if top model scores tie.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate models to persist in provenance.")
    parser.add_argument("--include-non-success", action="store_true", help="Allow non-success model rows in selection.")

    parser.add_argument("--rig-id", type=str, default=None, help="Filter by rig_id (registry source only).")
    parser.add_argument("--arena-id", type=str, default=None, help="Filter by arena_id (registry source only).")
    parser.add_argument("--camera-id", type=str, default=None, help="Filter by camera_id (registry source only).")
    parser.add_argument("--path-contains", type=str, default=None, help="Substring match on zarr_path (registry source only).")

    parser.add_argument("--config", type=str, default=None, help="Optional detect config path.")
    parser.add_argument("--conf", type=float, default=None, help="Optional detect confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="Optional detect IoU threshold override.")
    parser.add_argument("--max-det", type=int, default=None, help="Optional detect max_det override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional detect batch size override.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--write-raw-video-metadata",
        action="store_true",
        help="Write metadata-only raw_video attrs during detect.",
    )
    parser.add_argument(
        "--overwrite-raw-video-metadata",
        action="store_true",
        help="Overwrite existing metadata-only raw_video attrs during detect.",
    )

    parser.add_argument(
        "--scheduler",
        choices=["threads", "processes", "single-threaded"],
        default=None,
        help="Legacy option kept for compatibility; ignored by registry detect path.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Legacy option kept for compatibility; ignored by registry detect path.",
    )
    parser.add_argument(
        "--no-dask-progress",
        action="store_true",
        help="Legacy option kept for compatibility; ignored by registry detect path.",
    )

    parser.add_argument("--json", action="store_true", help="Emit JSON lines for plan/result rows.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/run_detections_batch or <root>/logs/run_detections_batch).",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging.")

    args = parser.parse_args(argv)

    inputs = _resolve_input_paths(args.paths, args.file_list or [])
    registry_path = (args.registry or Path("/nvme1/palette_registry.sqlite")).expanduser().resolve()

    skip_existing = not args.overwrite
    require_background = bool(args.require_background) and not bool(args.no_require_background)

    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, inputs)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"run_detections_batch_{run_id}.jsonl"
            logger = JsonLogger(log_path, run_id)
        except Exception as exc:
            fallback_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "palette" / "run_detections_batch"
            try:
                fallback_dir.mkdir(parents=True, exist_ok=True)
                log_path = fallback_dir / f"run_detections_batch_{run_id}.jsonl"
                logger = JsonLogger(log_path, run_id)
                print(
                    f"Warning: failed to initialize log dir {log_dir}; using fallback {fallback_dir} ({exc})",
                    file=sys.stderr,
                )
            except Exception as fallback_exc:
                logger = None
                print(
                    f"Warning: logging disabled (failed to initialize {log_dir} and fallback {fallback_dir}: {fallback_exc})",
                    file=sys.stderr,
                )

        if logger is not None:
            assert log_path is not None
            print(f"Log file: {log_path}")
            logger.log(
                "run_start",
                zarr="-",
                status="started",
                source=args.source,
                paths=[str(path) for path in inputs],
                recursive=bool(args.recursive),
                apply=bool(args.apply),
                dry_run=not bool(args.apply),
                overwrite=bool(args.overwrite),
                sql_prefilter=bool(args.source == "registry" and skip_existing),
                require_background=require_background,
                require_tuning=bool(args.require_tuning),
                registry=str(registry_path),
                set_id=args.set_id,
                require_unique=bool(args.require_unique),
                include_non_success=bool(args.include_non_success),
                top_k=int(args.top_k),
            )

    if args.source == "registry":
        if not registry_path.exists():
            print(f"Registry not found: {registry_path}", file=sys.stderr)
            if logger is not None:
                logger.log("run_end", zarr="-", status="failed", reason="registry_not_found")
                logger.close()
            return 1
        zarr_paths = _discover_analysis_zarrs_from_registry(
            registry_path=registry_path,
            scope_paths=inputs,
            rig_id=args.rig_id,
            arena_id=args.arena_id,
            camera_id=args.camera_id,
            path_contains=args.path_contains,
            skip_existing=skip_existing,
        )
    else:
        zarr_paths = _discover_analysis_zarrs(inputs, recursive=bool(args.recursive))

    if args.emit_paths:
        for p in zarr_paths:
            print(p)
        if logger is not None:
            logger.log("emit_paths", zarr="-", status="ok", count=len(zarr_paths))
            logger.close()
        return 0
    plans = _build_plans(
        zarr_paths,
        skip_existing=skip_existing,
        require_background=require_background,
        require_tuning=bool(args.require_tuning),
    )
    plans = _apply_registry_prereq(plans, registry_path=registry_path)

    if not args.apply:
        console = Console() if Console is not None else None
        if console is not None:
            console.rule("[bold yellow]Dry run[/bold yellow]")
            console.print("Add [cyan]--apply[/cyan] to run detection.")
        else:
            print("Dry run: add --apply to run detection.")

        if args.json:
            for plan in plans:
                print(json.dumps(_plan_payload(plan), sort_keys=True))
        else:
            _print_plan(plans)

        if logger is not None:
            for plan in plans:
                logger.log(
                    "detect_plan",
                    zarr=str(plan.zarr_path),
                    status=plan.status,
                    reason=plan.reason,
                    recording=str(plan.recording_dir),
                    video=str(plan.video_path) if plan.video_path else None,
                    detect_present=plan.detect_present,
                    background_present=plan.background_present,
                    tuning_present=plan.tuning_present,
                )
            counts = _counts_from_plans(plans)
            logger.log(
                "run_end",
                zarr="-",
                status="ok",
                ok=counts.get(STATUS_OK, 0),
                failed=0,
                skipped=counts.get(STATUS_SKIPPED, 0),
                missing=counts.get(STATUS_MISSING, 0),
                dry_run=True,
            )
            logger.close()
        return 0

    ok = 0
    failed = 0
    skipped = 0
    missing = 0

    for plan in plans:
        if plan.status == STATUS_MISSING:
            missing += 1
            if args.json:
                print(
                    json.dumps(
                        {
                            "recording": str(plan.recording_dir),
                            "zarr": str(plan.zarr_path),
                            "status": STATUS_MISSING,
                            "reason": plan.reason,
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(f"Skipping (missing prerequisites): {plan.zarr_path} ({plan.reason})")
            if logger is not None:
                logger.log("detect_skipped", zarr=str(plan.zarr_path), status=STATUS_MISSING, reason=plan.reason)
            continue

        if plan.status == STATUS_SKIPPED and skip_existing:
            skipped += 1
            if args.json:
                print(
                    json.dumps(
                        {
                            "recording": str(plan.recording_dir),
                            "zarr": str(plan.zarr_path),
                            "status": STATUS_SKIPPED,
                            "reason": plan.reason,
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(f"Skipping (detect exists): {plan.zarr_path}")
            if logger is not None:
                logger.log("detect_skipped", zarr=str(plan.zarr_path), status=STATUS_SKIPPED, reason=plan.reason)
            continue

        result = run_detect_with_registry_model(
            recording_dir=plan.recording_dir,
            video=plan.video_path,
            output=plan.zarr_path,
            registry=registry_path,
            set_id=args.set_id,
            require_unique=bool(args.require_unique),
            top_k=int(args.top_k),
            include_non_success=bool(args.include_non_success),
            dry_run=False,
            config=args.config,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            cpu=bool(args.cpu),
            write_raw_video_metadata=bool(args.write_raw_video_metadata),
            overwrite_raw_video_metadata=bool(args.overwrite_raw_video_metadata),
        )

        if result.ok:
            ok += 1
            if args.json:
                print(
                    json.dumps(
                        {
                            "recording": result.recording_dir,
                            "zarr": result.output_zarr,
                            "status": STATUS_OK,
                            "detect_run": result.detect_run,
                            "selected_model": result.selected_model_path,
                            "selected_run_id": result.selected_run_id,
                            "selected_set_id": result.selected_set_id,
                        },
                        sort_keys=True,
                    )
                )
            if logger is not None:
                logger.log(
                    "detect_ok",
                    zarr=result.output_zarr,
                    status=STATUS_OK,
                    detect_run=result.detect_run,
                    selected_model=result.selected_model_path,
                    selected_run_id=result.selected_run_id,
                    selected_set_id=result.selected_set_id,
                    resolved_at_utc=result.resolved_at_utc,
                )
            continue

        failed += 1
        if args.json:
            print(
                json.dumps(
                    {
                        "recording": result.recording_dir,
                        "zarr": result.output_zarr,
                        "status": STATUS_FAILED,
                        "reason": result.reason,
                        "error": result.error,
                        "remediation": result.remediation,
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"Detection failed for {result.output_zarr}: {result.reason} ({result.error})")

        if logger is not None:
            logger.log(
                "detect_failed",
                zarr=result.output_zarr,
                status=STATUS_FAILED,
                reason=result.reason,
                error=result.error,
                remediation=result.remediation,
                selected_model=result.selected_model_path,
                selected_run_id=result.selected_run_id,
                selected_set_id=result.selected_set_id,
            )

    if logger is not None:
        logger.log(
            "run_end",
            zarr="-",
            status=(STATUS_OK if failed == 0 else STATUS_FAILED),
            reason=(None if failed == 0 else "batch_failures_detected"),
            ok=ok,
            failed=failed,
            skipped=skipped,
            missing=missing,
            dry_run=False,
        )
        logger.close()

    if not args.json:
        print("\nSummary:")
        print(f"  ok: {ok}")
        print(f"  failed: {failed}")
        print(f"  skipped: {skipped}")
        print(f"  missing: {missing}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
