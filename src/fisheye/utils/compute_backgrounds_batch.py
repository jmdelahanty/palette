import argparse
import concurrent.futures
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import zarr

from fisheye.preprocessing.background import compute_background

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
class BackgroundPlan:
    recording_dir: Path
    h5_path: Path
    zarr_path: Path
    camera_id: Optional[str]
    status: str
    reason: Optional[str] = None
    background_present: bool = False


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
        return Path(env_root) / "compute_backgrounds_batch"
    if roots:
        return roots[0] / "logs" / "compute_backgrounds_batch"
    return Path.cwd() / "logs" / "compute_backgrounds_batch"


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


def _compute_background_worker(
    zarr_path: str,
    sample_size: Optional[int],
    seed: Optional[int],
    method: Optional[str],
    compute_full: bool,
    compute_ds: bool,
    config_path: str,
    quiet: bool,
) -> Dict:
    if quiet and Console is not None:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            console = Console(file=devnull, force_terminal=False)
            return compute_background(
                zarr_path,
                sample_size=sample_size,
                seed=seed,
                method=method,
                compute_full=compute_full,
                compute_ds=compute_ds,
                config_path=config_path,
                console=console,
            )
    return compute_background(
        zarr_path,
        sample_size=sample_size,
        seed=seed,
        method=method,
        compute_full=compute_full,
        compute_ds=compute_ds,
        config_path=config_path,
        console=None,
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


def _has_background(zarr_path: Path) -> bool:
    if not zarr_path.exists():
        return False
    root = zarr.open(str(zarr_path), mode="r")
    bg = root.get("background_runs")
    if bg is None:
        return False
    latest = bg.attrs.get("latest")
    return bool(latest)


def _build_plans(roots: List[Path], recursive: bool, skip_existing: bool) -> List[BackgroundPlan]:
    plans: List[BackgroundPlan] = []
    for h5_path in _iter_h5(roots, recursive):
        recording_dir = h5_path.parent.parent
        zarr_path = recording_dir / "zarr" / f"{h5_path.stem}.zarr"
        camera_id = _read_camera_id(h5_path)
        if not zarr_path.exists():
            plans.append(
                BackgroundPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=camera_id,
                    status="missing",
                    reason="zarr missing",
                )
            )
            continue
        bg_present = _has_background(zarr_path)
        if skip_existing and bg_present:
            plans.append(
                BackgroundPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=camera_id,
                    status="skipped",
                    reason="background already present",
                    background_present=True,
                )
            )
            continue
        plans.append(
            BackgroundPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="ok",
                background_present=bg_present,
            )
        )
    return plans


def _print_plan(plans: List[BackgroundPlan]) -> None:
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
        description="Batch compute background models for recordings.",
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
        help="Run background computation (default: dry-run).",
    )
    apply_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned background runs without computing (default behavior).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Compute background even if a background run already exists.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of frames to sample (-1 for all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for frame sampling.",
    )
    parser.add_argument(
        "--method",
        choices=["mode", "median"],
        default=None,
        help="Background computation method.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full resolution background.",
    )
    parser.add_argument(
        "--skip-ds",
        action="store_true",
        help="Skip downsampled background.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="Path to pipeline config file (used when params are omitted).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of recordings to process in parallel (default: 1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON lines for each plan/result.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/compute_backgrounds_batch or <recordings_root>/logs/compute_backgrounds_batch).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable JSONL logging.",
    )

    args = parser.parse_args(argv)
    roots = _resolve_root(args.paths)
    skip_existing = not args.overwrite

    if args.workers < 1:
        print(f"--workers must be >= 1 (got {args.workers})")
        return 2

    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"compute_backgrounds_batch_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            roots=[str(root) for root in roots],
            recursive=bool(args.recursive),
            apply=bool(args.apply),
            dry_run=not bool(args.apply),
            overwrite=bool(args.overwrite),
            sample_size=args.sample_size,
            seed=args.seed,
            method=args.method,
            skip_full=bool(args.skip_full),
            skip_ds=bool(args.skip_ds),
            config=args.config,
            json=bool(args.json),
            workers=int(args.workers),
        )

    plans = _build_plans(roots, args.recursive, skip_existing=skip_existing)
    if not args.apply:
        console = Console() if Console is not None else None
        if console is not None:
            console.rule("[bold yellow]Dry run[/bold yellow]")
            console.print("Add [cyan]--apply[/cyan] to compute backgrounds.")
        else:
            print("Dry run: add --apply to compute backgrounds.")
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
                        "background_plan",
                        recording=str(plan.recording_dir),
                        zarr=str(plan.zarr_path),
                        camera_id=plan.camera_id,
                        status=plan.status,
                        reason=plan.reason,
                        background_present=plan.background_present,
                    )
        else:
            _print_plan(plans)
            if logger is not None:
                for plan in plans:
                    logger.log(
                        "background_plan",
                        recording=str(plan.recording_dir),
                        zarr=str(plan.zarr_path),
                        camera_id=plan.camera_id,
                        status=plan.status,
                        reason=plan.reason,
                        background_present=plan.background_present,
                    )
        if logger is not None:
            logger.log("run_end", ok=0, failed=0, skipped=0, missing=0, dry_run=True)
            logger.close()
        return 0

    console = Console() if Console is not None else None
    if console is not None and args.workers > 4 and not args.skip_full:
        console.print(
            "[yellow]Warning:[/yellow] --workers > 4 with full-resolution backgrounds "
            "can be IO- and memory-heavy. Consider --skip-full or fewer workers."
        )
    elif args.workers > 4 and not args.skip_full:
        print(
            "Warning: --workers > 4 with full-resolution backgrounds can be IO- and memory-heavy. "
            "Consider --skip-full or fewer workers."
        )

    ok = 0
    failed = 0
    skipped = 0
    missing = 0

    runnable_plans: List[BackgroundPlan] = []
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            if args.json:
                print(json.dumps({"status": "missing", "zarr": str(plan.zarr_path)}))
            else:
                print(f"Skipping (missing zarr): {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "background_skipped",
                    zarr=str(plan.zarr_path),
                    status="missing",
                    reason="zarr missing",
                )
            continue
        if plan.status == "skipped" and skip_existing:
            skipped += 1
            if args.json:
                print(json.dumps({"status": "skipped", "zarr": str(plan.zarr_path)}))
            else:
                print(f"Skipping (background exists): {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "background_skipped",
                    zarr=str(plan.zarr_path),
                    status="skipped",
                    reason="background already present",
                )
            continue
        runnable_plans.append(plan)

    progress = _progress(console if not args.json else None, total=len(runnable_plans))
    if args.workers == 1:
        if progress is None:
            for plan in runnable_plans:
                try:
                    results = compute_background(
                        str(plan.zarr_path),
                        sample_size=args.sample_size,
                        seed=args.seed,
                        method=args.method,
                        compute_full=not args.skip_full,
                        compute_ds=not args.skip_ds,
                        config_path=args.config,
                        console=console,
                    )
                    ok += 1
                    if args.json:
                        print(json.dumps({"status": "ok", "zarr": str(plan.zarr_path)}))
                    if logger is not None:
                        logger.log(
                            "background_ok",
                            zarr=str(plan.zarr_path),
                            results=results,
                        )
                except Exception as exc:
                    failed += 1
                    if args.json:
                        print(json.dumps({"status": "failed", "zarr": str(plan.zarr_path), "error": str(exc)}))
                    else:
                        print(f"Background failed for {plan.zarr_path}: {exc}")
                    if logger is not None:
                        logger.log(
                            "background_failed",
                            zarr=str(plan.zarr_path),
                            error=str(exc),
                        )
        else:
            with progress:
                task = progress.add_task("Computing backgrounds", total=len(runnable_plans))
                for plan in runnable_plans:
                    try:
                        results = compute_background(
                            str(plan.zarr_path),
                            sample_size=args.sample_size,
                            seed=args.seed,
                            method=args.method,
                            compute_full=not args.skip_full,
                            compute_ds=not args.skip_ds,
                            config_path=args.config,
                            console=console,
                        )
                        ok += 1
                        if args.json:
                            print(json.dumps({"status": "ok", "zarr": str(plan.zarr_path)}))
                        if logger is not None:
                            logger.log(
                                "background_ok",
                                zarr=str(plan.zarr_path),
                                results=results,
                            )
                    except Exception as exc:
                        failed += 1
                        if args.json:
                            print(
                                json.dumps({"status": "failed", "zarr": str(plan.zarr_path), "error": str(exc)})
                            )
                        else:
                            print(f"Background failed for {plan.zarr_path}: {exc}")
                        if logger is not None:
                            logger.log(
                                "background_failed",
                                zarr=str(plan.zarr_path),
                                error=str(exc),
                            )
                    progress.advance(task)
    else:
        quiet = True
        if progress is None:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_map = {
                    executor.submit(
                        _compute_background_worker,
                        str(plan.zarr_path),
                        args.sample_size,
                        args.seed,
                        args.method,
                        not args.skip_full,
                        not args.skip_ds,
                        args.config,
                        quiet,
                    ): plan
                    for plan in runnable_plans
                }
                for future in concurrent.futures.as_completed(future_map):
                    plan = future_map[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        failed += 1
                        if args.json:
                            print(
                                json.dumps(
                                    {"status": "failed", "zarr": str(plan.zarr_path), "error": str(exc)}
                                )
                            )
                        else:
                            print(f"Background failed for {plan.zarr_path}: {exc}")
                        if logger is not None:
                            logger.log(
                                "background_failed",
                                zarr=str(plan.zarr_path),
                                error=str(exc),
                            )
                        continue

                    ok += 1
                    if args.json:
                        print(json.dumps({"status": "ok", "zarr": str(plan.zarr_path)}))
                    if logger is not None:
                        logger.log(
                            "background_ok",
                            zarr=str(plan.zarr_path),
                            results=results,
                        )
        else:
            with progress:
                task = progress.add_task("Computing backgrounds", total=len(runnable_plans))
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                    future_map = {
                        executor.submit(
                            _compute_background_worker,
                            str(plan.zarr_path),
                            args.sample_size,
                            args.seed,
                            args.method,
                            not args.skip_full,
                            not args.skip_ds,
                            args.config,
                            quiet,
                        ): plan
                        for plan in runnable_plans
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        plan = future_map[future]
                        try:
                            results = future.result()
                        except Exception as exc:
                            failed += 1
                            if args.json:
                                print(
                                    json.dumps(
                                        {"status": "failed", "zarr": str(plan.zarr_path), "error": str(exc)}
                                    )
                                )
                            else:
                                print(f"Background failed for {plan.zarr_path}: {exc}")
                            if logger is not None:
                                logger.log(
                                    "background_failed",
                                    zarr=str(plan.zarr_path),
                                    error=str(exc),
                                )
                        else:
                            ok += 1
                            if args.json:
                                print(json.dumps({"status": "ok", "zarr": str(plan.zarr_path)}))
                            if logger is not None:
                                logger.log(
                                    "background_ok",
                                    zarr=str(plan.zarr_path),
                                    results=results,
                                )
                        progress.advance(task)

    print("\nSummary")
    print(f"  ok: {ok}")
    print(f"  failed: {failed}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    if logger is not None:
        logger.log("run_end", ok=ok, failed=failed, skipped=skipped, missing=missing, dry_run=False)
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
