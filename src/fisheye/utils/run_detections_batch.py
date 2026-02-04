import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import zarr

from fisheye.detection.detect_traditional import detect_fish

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore
    Progress = None  # type: ignore
    SpinnerColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    BarColumn = None  # type: ignore
    TimeElapsedColumn = None  # type: ignore
    TimeRemainingColumn = None  # type: ignore


@dataclass
class DetectPlan:
    recording_dir: Path
    h5_path: Path
    zarr_path: Path
    camera_id: Optional[str]
    status: str
    reason: Optional[str] = None
    detect_present: bool = False
    background_present: bool = False
    tuning_present: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, event: str, **fields: object) -> None:
        payload = {"event": event, "ts_utc": _utc_now(), "run_id": self.run_id}
        payload.update(fields)
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _normalize_attr(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    text = _normalize_attr(ipc_source_name)
    if text is None:
        return None
    match = re.search(r"cam_(\d+)", text)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", text)
    return digits[-1] if digits else None


def _read_camera_id(h5_path: Path) -> Optional[str]:
    with h5py.File(h5_path, "r") as h5:
        root = h5.attrs
        if "camera_id" in root:
            cam = _normalize_attr(root.get("camera_id"))
            if cam:
                return cam
        ipc = _normalize_attr(root.get("ipc_source_name"))
        return _derive_camera_id(ipc)


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "run_detections_batch"
    if roots:
        return roots[0] / "logs" / "run_detections_batch"
    return Path.cwd() / "logs" / "run_detections_batch"


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{os.getpid()}"


def _progress(console: Optional[Console], total: int):
    if console is None or Progress is None:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _iter_h5(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            if path.suffix.lower() in {".h5", ".hdf5"}:
                yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("raw/*.h5")
            yield from path.rglob("raw/*.hdf5")
        else:
            yield from path.glob("*/raw/*.h5")
            yield from path.glob("*/raw/*.hdf5")


def _has_background(root: zarr.Group) -> bool:
    bg = root.get("background_runs")
    if bg is None:
        return False
    latest = bg.attrs.get("latest")
    return bool(latest)


def _has_detection(root: zarr.Group) -> bool:
    detect = root.get("detect_runs")
    if detect is None:
        return False
    latest = detect.attrs.get("latest")
    return bool(latest)


def _has_detection_tuning(root: zarr.Group) -> bool:
    meta = root.get("analysis_metadata")
    if meta is None:
        return False
    return "detection_tuning" in meta.attrs


def _has_images_ds(root: zarr.Group) -> bool:
    raw = root.get("raw_video")
    if raw is None:
        return False
    return "images_ds" in raw


def _build_plans(
    roots: List[Path],
    recursive: bool,
    skip_existing: bool,
    require_background: bool,
    require_tuning: bool,
) -> List[DetectPlan]:
    plans: List[DetectPlan] = []
    for h5_path in _iter_h5(roots, recursive):
        recording_dir = h5_path.parent.parent
        zarr_path = recording_dir / "zarr" / f"{h5_path.stem}.zarr"
        camera_id = _read_camera_id(h5_path)
        if not zarr_path.exists():
            plans.append(
                DetectPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=camera_id,
                    status="missing",
                    reason="zarr missing",
                )
            )
            continue
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception as exc:
            plans.append(
                DetectPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=camera_id,
                    status="missing",
                    reason=f"zarr open failed: {exc}",
                )
            )
            continue

        detect_present = _has_detection(root)
        background_present = _has_background(root)
        tuning_present = _has_detection_tuning(root)
        images_ds_present = _has_images_ds(root)

        if require_background and not background_present:
            plans.append(
                DetectPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=camera_id,
                    status="missing",
                    reason="background missing",
                    detect_present=detect_present,
                    background_present=background_present,
                    tuning_present=tuning_present,
                )
            )
            continue
        if not images_ds_present:
            plans.append(
                DetectPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=camera_id,
                    status="missing",
                    reason="raw_video/images_ds missing",
                    detect_present=detect_present,
                    background_present=background_present,
                    tuning_present=tuning_present,
                )
            )
            continue
        if require_tuning and not tuning_present:
            plans.append(
                DetectPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=camera_id,
                    status="missing",
                    reason="detection tuning missing",
                    detect_present=detect_present,
                    background_present=background_present,
                    tuning_present=tuning_present,
                )
            )
            continue
        if skip_existing and detect_present:
            plans.append(
                DetectPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=camera_id,
                    status="skipped",
                    reason="detect run already present",
                    detect_present=detect_present,
                    background_present=background_present,
                    tuning_present=tuning_present,
                )
            )
            continue
        plans.append(
            DetectPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="ok",
                detect_present=detect_present,
                background_present=background_present,
                tuning_present=tuning_present,
            )
        )
    return plans


def _print_plan(plans: List[DetectPlan]) -> None:
    counts = {"ok": 0, "skipped": 0, "missing": 0}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
        print(f"Recording: {plan.recording_dir.name}")
        print(f"  camera_id: {plan.camera_id or 'unknown'}")
        print(f"  zarr: {plan.zarr_path}")
        print(f"  status: {plan.status}")
        if plan.reason:
            print(f"  reason: {plan.reason}")
    print("\nSummary:")
    print(f"  ok: {counts.get('ok', 0)}")
    print(f"  skipped: {counts.get('skipped', 0)}")
    print(f"  missing: {counts.get('missing', 0)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch run detection on recordings (blob-based, zarr-backed).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording root(s) to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for recordings under each root.",
    )
    apply_group = parser.add_mutually_exclusive_group()
    apply_group.add_argument(
        "--apply",
        action="store_true",
        help="Run detections (default: dry-run).",
    )
    apply_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned detections without running (default behavior).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run detection even if a detect run already exists.",
    )
    parser.add_argument(
        "--require-background",
        action="store_true",
        help="Require a background run to exist before detecting (default).",
    )
    parser.add_argument(
        "--no-require-background",
        action="store_true",
        help="Allow detection to run even if background is missing.",
    )
    parser.add_argument(
        "--require-tuning",
        action="store_true",
        help="Skip recordings without detection_tuning in analysis_metadata.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fisheye/default.yaml",
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["threads", "processes", "single-threaded"],
        default="processes",
        help="Dask scheduler to use (passed to detection stage).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional worker count hint for the detection scheduler.",
    )
    parser.add_argument(
        "--no-dask-progress",
        action="store_true",
        help="Disable the Dask progress bar (recommended for batch runs).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON lines for each plan/result.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/run_detections_batch or <recordings_root>/logs/run_detections_batch).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable JSONL logging.",
    )

    args = parser.parse_args(argv)
    roots = _resolve_root(args.paths)
    skip_existing = not args.overwrite
    require_background = bool(args.require_background) and not bool(args.no_require_background)

    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_detections_batch_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            roots=[str(root) for root in roots],
            recursive=bool(args.recursive),
            apply=bool(args.apply),
            dry_run=not bool(args.apply),
            overwrite=bool(args.overwrite),
            require_background=require_background,
            require_tuning=bool(args.require_tuning),
            config=args.config,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            json=bool(args.json),
        )

    plans = _build_plans(
        roots,
        args.recursive,
        skip_existing=skip_existing,
        require_background=require_background,
        require_tuning=bool(args.require_tuning),
    )

    if not args.apply:
        console = Console() if Console is not None else None
        if console is not None:
            console.rule("[bold yellow]Dry run[/bold yellow]")
            console.print("Add [cyan]--apply[/cyan] to run detections.")
        else:
            print("Dry run: add --apply to run detections.")
        if args.json:
            for plan in plans:
                print(
                    json.dumps(
                        {
                            "recording": plan.recording_dir.name,
                            "zarr": str(plan.zarr_path),
                            "camera_id": plan.camera_id,
                            "status": plan.status,
                            "reason": plan.reason,
                        }
                    )
                )
                if logger is not None:
                    logger.log(
                        "detect_plan",
                        recording=str(plan.recording_dir),
                        zarr=str(plan.zarr_path),
                        camera_id=plan.camera_id,
                        status=plan.status,
                        reason=plan.reason,
                        detect_present=plan.detect_present,
                        background_present=plan.background_present,
                        tuning_present=plan.tuning_present,
                    )
        else:
            _print_plan(plans)
            if logger is not None:
                for plan in plans:
                    logger.log(
                        "detect_plan",
                        recording=str(plan.recording_dir),
                        zarr=str(plan.zarr_path),
                        camera_id=plan.camera_id,
                        status=plan.status,
                        reason=plan.reason,
                        detect_present=plan.detect_present,
                        background_present=plan.background_present,
                        tuning_present=plan.tuning_present,
                    )
        if logger is not None:
            logger.log("run_end", ok=0, failed=0, skipped=0, missing=0, dry_run=True)
            logger.close()
        return 0

    ok = 0
    failed = 0
    skipped = 0
    missing = 0

    runnable_plans: List[DetectPlan] = []
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            if args.json:
                print(json.dumps({"status": "missing", "zarr": str(plan.zarr_path)}))
            else:
                print(f"Skipping (missing prerequisites): {plan.zarr_path} ({plan.reason})")
            if logger is not None:
                logger.log(
                    "detect_skipped",
                    zarr=str(plan.zarr_path),
                    status="missing",
                    reason=plan.reason,
                )
            continue
        if plan.status == "skipped" and skip_existing:
            skipped += 1
            if args.json:
                print(json.dumps({"status": "skipped", "zarr": str(plan.zarr_path)}))
            else:
                print(f"Skipping (detect exists): {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "detect_skipped",
                    zarr=str(plan.zarr_path),
                    status="skipped",
                    reason="detect run already present",
                )
            continue
        runnable_plans.append(plan)

    console = Console() if Console is not None else None
    progress = _progress(console if not args.json else None, total=len(runnable_plans))

    if progress is None:
        for plan in runnable_plans:
            try:
                result = detect_fish(
                    zarr_path=str(plan.zarr_path),
                    config_path=args.config,
                    scheduler=args.scheduler,
                    num_workers=args.num_workers,
                    console=console,
                    show_progress=not args.no_dask_progress,
                )
                ok += 1
                if args.json:
                    print(json.dumps({"status": "ok", "zarr": str(plan.zarr_path)}))
                if logger is not None:
                    logger.log(
                        "detect_ok",
                        zarr=str(plan.zarr_path),
                        results=result,
                    )
            except Exception as exc:
                failed += 1
                if args.json:
                    print(json.dumps({"status": "failed", "zarr": str(plan.zarr_path), "error": str(exc)}))
                else:
                    print(f"Detection failed for {plan.zarr_path}: {exc}")
                if logger is not None:
                    logger.log(
                        "detect_failed",
                        zarr=str(plan.zarr_path),
                        error=str(exc),
                    )
    else:
        with progress:
            task = progress.add_task("Running detections", total=len(runnable_plans))
            for plan in runnable_plans:
                try:
                    result = detect_fish(
                        zarr_path=str(plan.zarr_path),
                        config_path=args.config,
                        scheduler=args.scheduler,
                        num_workers=args.num_workers,
                        console=console,
                        show_progress=not args.no_dask_progress,
                    )
                    ok += 1
                    if args.json:
                        print(json.dumps({"status": "ok", "zarr": str(plan.zarr_path)}))
                    if logger is not None:
                        logger.log(
                            "detect_ok",
                            zarr=str(plan.zarr_path),
                            results=result,
                        )
                except Exception as exc:
                    failed += 1
                    if args.json:
                        print(json.dumps({"status": "failed", "zarr": str(plan.zarr_path), "error": str(exc)}))
                    else:
                        print(f"Detection failed for {plan.zarr_path}: {exc}")
                    if logger is not None:
                        logger.log(
                            "detect_failed",
                            zarr=str(plan.zarr_path),
                            error=str(exc),
                        )
                progress.advance(task)

    if logger is not None:
        logger.log(
            "run_end",
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
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
