import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import yaml
import zarr

from fisheye.detection.detect_keypoints_traditional import detect_keypoints
from fisheye.detection.detect_keypoints_yolo import detect_keypoints_yolo
from fisheye.refinement.refine_keypoints import create_refined_keypoint_run

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
class KeypointPlan:
    recording_dir: Path
    h5_path: Path
    zarr_path: Path
    camera_id: Optional[str]
    status: str
    reason: Optional[str] = None
    keypoints_present: bool = False
    crop_present: bool = False
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
        return Path(env_root) / "run_keypoints_batch"
    if roots:
        return roots[0] / "logs" / "run_keypoints_batch"
    return Path.cwd() / "logs" / "run_keypoints_batch"


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


def _iter_zarr(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_file() and path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


def _is_zarr_path(path: Path) -> bool:
    return path.suffix == ".zarr"


def _load_paths_file(path: Path) -> List[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    items: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _infer_recording_dir(zarr_path: Path) -> Path:
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    return zarr_path.parent


def _find_h5(recording_dir: Path, stem: str) -> Optional[Path]:
    for suffix in (".h5", ".hdf5"):
        candidate = recording_dir / "raw" / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _has_keypoints(root: zarr.Group) -> bool:
    kp = root.get("keypoints_runs")
    if kp is None:
        return False
    return bool(kp.attrs.get("latest"))


def _has_crop(root: zarr.Group) -> bool:
    crop = root.get("crop_runs")
    if crop is None:
        return False
    return bool(crop.attrs.get("latest"))


def _has_background(root: zarr.Group) -> bool:
    bg = root.get("background_runs")
    if bg is None:
        return False
    return bool(bg.attrs.get("latest"))


def _has_keypoint_tuning(root: zarr.Group) -> bool:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return False
    return "keypoint_tuning" in analysis.attrs


def _latest_keypoints_run(zarr_path: Path) -> Optional[str]:
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception:
        return None
    group = root.get("keypoints_runs")
    if group is None:
        return None
    return group.attrs.get("latest")


def _keypoints_total_rois(zarr_path: Path, run_name: Optional[str]) -> Optional[int]:
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception:
        return None
    group = root.get("keypoints_runs")
    if group is None:
        return None
    resolved = run_name or group.attrs.get("latest")
    if not resolved or resolved not in group:
        return None
    run_group = group[resolved]
    if "keypoints_roi" in run_group:
        return int(run_group["keypoints_roi"].shape[0])
    summary = run_group.attrs.get("summary_statistics", {})
    if isinstance(summary, dict) and "total_rois" in summary:
        try:
            return int(summary["total_rois"])
        except (TypeError, ValueError):
            return None
    return None


def _plan_from_zarr(
    *,
    zarr_path: Path,
    recording_dir: Path,
    h5_path: Path,
    camera_id: Optional[str],
    skip_existing: bool,
    require_crop: bool,
    require_background: bool,
    require_tuning: bool,
    refine_only: bool,
) -> KeypointPlan:
    if not zarr_path.exists():
        return KeypointPlan(
            recording_dir=recording_dir,
            h5_path=h5_path,
            zarr_path=zarr_path,
            camera_id=camera_id,
            status="missing",
            reason="zarr missing",
        )

    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception as exc:
        return KeypointPlan(
            recording_dir=recording_dir,
            h5_path=h5_path,
            zarr_path=zarr_path,
            camera_id=camera_id,
            status="missing",
            reason=f"zarr open failed: {exc}",
        )

    crop_present = _has_crop(root)
    background_present = _has_background(root)
    keypoints_present = _has_keypoints(root)
    tuning_present = _has_keypoint_tuning(root)

    if refine_only:
        if not keypoints_present:
            return KeypointPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="missing",
                reason="keypoints missing",
                crop_present=crop_present,
                background_present=background_present,
                keypoints_present=keypoints_present,
                tuning_present=tuning_present,
            )
    else:
        if require_crop and not crop_present:
            return KeypointPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="missing",
                reason="crop missing",
                crop_present=crop_present,
                background_present=background_present,
                keypoints_present=keypoints_present,
                tuning_present=tuning_present,
            )
        if require_background and not background_present:
            return KeypointPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="missing",
                reason="background missing",
                crop_present=crop_present,
                background_present=background_present,
                keypoints_present=keypoints_present,
                tuning_present=tuning_present,
            )
        if require_tuning and not tuning_present:
            return KeypointPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="missing",
                reason="keypoint_tuning missing",
                crop_present=crop_present,
                background_present=background_present,
                keypoints_present=keypoints_present,
                tuning_present=tuning_present,
            )
        if skip_existing and keypoints_present:
            return KeypointPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="skipped",
                reason="keypoints already present",
                crop_present=crop_present,
                background_present=background_present,
                keypoints_present=keypoints_present,
                tuning_present=tuning_present,
            )

    return KeypointPlan(
        recording_dir=recording_dir,
        h5_path=h5_path,
        zarr_path=zarr_path,
        camera_id=camera_id,
        status="ok",
        crop_present=crop_present,
        background_present=background_present,
        keypoints_present=keypoints_present,
        tuning_present=tuning_present,
    )


def _build_plans(
    roots: List[Path],
    recursive: bool,
    skip_existing: bool,
    require_crop: bool,
    require_background: bool,
    require_tuning: bool,
    refine_only: bool,
) -> List[KeypointPlan]:
    plans: List[KeypointPlan] = []
    for h5_path in _iter_h5(roots, recursive):
        recording_dir = h5_path.parent.parent
        zarr_path = recording_dir / "zarr" / f"{h5_path.stem}.zarr"
        camera_id = _read_camera_id(h5_path)
        plans.append(
            _plan_from_zarr(
                zarr_path=zarr_path,
                recording_dir=recording_dir,
                h5_path=h5_path,
                camera_id=camera_id,
                skip_existing=skip_existing,
                require_crop=require_crop,
                require_background=require_background,
                require_tuning=require_tuning,
                refine_only=refine_only,
            )
        )
    return plans


def _build_plans_from_zarr(
    zarr_paths: Iterable[Path],
    skip_existing: bool,
    require_crop: bool,
    require_background: bool,
    require_tuning: bool,
    refine_only: bool,
) -> List[KeypointPlan]:
    plans: List[KeypointPlan] = []
    for zarr_path in zarr_paths:
        recording_dir = _infer_recording_dir(zarr_path)
        h5_path = _find_h5(recording_dir, zarr_path.stem)
        camera_id = _read_camera_id(h5_path) if h5_path else None
        plans.append(
            _plan_from_zarr(
                zarr_path=zarr_path,
                recording_dir=recording_dir,
                h5_path=h5_path or (recording_dir / "raw" / f"{zarr_path.stem}.h5"),
                camera_id=camera_id,
                skip_existing=skip_existing,
                require_crop=require_crop,
                require_background=require_background,
                require_tuning=require_tuning,
                refine_only=refine_only,
            )
        )
    return plans


def _print_plan(plans: List[KeypointPlan]) -> None:
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


def _load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _resolve_method(config: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override.lower()
    method = config.get("keypoints", {}).get("method", "traditional")
    return str(method).lower()


def _run_traditional(
    zarr_path: str,
    config: Dict[str, Any],
    scheduler: Optional[str],
    num_workers: Optional[int],
    quiet: bool,
    show_progress: bool,
) -> Dict[str, Any]:
    if quiet and Console is not None:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            console = Console(file=devnull, force_terminal=False)
            return detect_keypoints(
                zarr_path=zarr_path,
                config=config,
                scheduler=scheduler,
                num_workers=num_workers,
                console=console,
                show_progress=show_progress,
            )
    return detect_keypoints(
        zarr_path=zarr_path,
        config=config,
        scheduler=scheduler,
        num_workers=num_workers,
        console=None,
        show_progress=show_progress,
    )


def _run_yolo(zarr_path: str, config: Dict[str, Any], quiet: bool) -> str:
    params = config.get("keypoints", {})
    model_path = params.get("model") or params.get("model_path")
    if not model_path:
        raise ValueError("YOLO keypoints require 'model' or 'model_path' in config.")
    if quiet and Console is not None:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            console = Console(file=devnull, force_terminal=False)
            return detect_keypoints_yolo(
                zarr_path=zarr_path,
                model_path=model_path,
                run_name=params.get("run_name"),
                crop_run=params.get("crop_run"),
                batch_size=params.get("batch_size", 256),
                device=params.get("device"),
                imgsz=params.get("imgsz"),
                conf=params.get("conf", 0.25),
                iou=params.get("iou", 0.5),
                max_det=params.get("max_det", 1),
                mask_threshold=params.get("mask_threshold", 0.5),
                verbose=params.get("verbose", False),
                console=console,
            )
    return detect_keypoints_yolo(
        zarr_path=zarr_path,
        model_path=model_path,
        run_name=params.get("run_name"),
        crop_run=params.get("crop_run"),
        batch_size=params.get("batch_size", 256),
        device=params.get("device"),
        imgsz=params.get("imgsz"),
        conf=params.get("conf", 0.25),
        iou=params.get("iou", 0.5),
        max_det=params.get("max_det", 1),
        mask_threshold=params.get("mask_threshold", 0.5),
        verbose=params.get("verbose", False),
        console=None,
    )


def _run_refine(
    zarr_path: str,
    config: Dict[str, Any],
    keypoint_run: Optional[str],
    quiet: bool,
) -> str:
    if quiet and Console is not None:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            console = Console(file=devnull, force_terminal=False)
            return create_refined_keypoint_run(
                zarr_path,
                keypoint_run=keypoint_run,
                config=config,
                console=console,
                command="run_keypoints_batch --refine",
            )
    return create_refined_keypoint_run(
        zarr_path,
        keypoint_run=keypoint_run,
        config=config,
        console=None,
        command="run_keypoints_batch --refine",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch run keypoint detection on recordings.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots, h5 paths, or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr or h5 path per line (comments with # allowed).",
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
        help="Run keypoint detection (default: dry-run).",
    )
    apply_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned keypoint runs without computing (default behavior).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run keypoints even if a keypoints run already exists.",
    )
    parser.add_argument(
        "--require-background",
        action="store_true",
        help="Require a background run to exist before running keypoints (default).",
    )
    parser.add_argument(
        "--no-require-background",
        action="store_true",
        help="Allow keypoints to run even if background is missing.",
    )
    parser.add_argument(
        "--require-crop",
        action="store_true",
        help="Require a crop run to exist before running keypoints (default).",
    )
    parser.add_argument(
        "--no-require-crop",
        action="store_true",
        help="Allow keypoints to run even if crops are missing.",
    )
    parser.add_argument(
        "--require-tuning",
        action="store_true",
        help="Skip recordings without keypoint_tuning in analysis_metadata.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fisheye/default.yaml",
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--method",
        choices=["traditional", "yolo"],
        help="Override keypoint method from config.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["threads", "processes", "single-threaded", "distributed"],
        default=None,
        help="Dask scheduler to use for traditional keypoints (defaults to config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional worker count hint for the keypoint scheduler.",
    )
    parser.add_argument(
        "--dask-progress",
        action="store_true",
        help="Enable per-chunk progress bars (disabled by default for batch runs).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-recording console output (recommended for batch runs).",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Run refine_keypoints immediately after keypoints detection.",
    )
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Refine existing keypoints runs without re-running keypoint detection.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON lines for each plan/result.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/run_keypoints_batch or <recordings_root>/logs/run_keypoints_batch).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable JSONL logging.",
    )
    parser.set_defaults(require_background=True, require_crop=True)

    args = parser.parse_args(argv)
    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    explicit_paths: List[Path] = []
    if args.paths:
        explicit_paths.extend(args.paths)
    if file_list_paths:
        explicit_paths.extend(file_list_paths)

    explicit_zarrs: List[Path] = []
    roots: List[Path] = []
    if explicit_paths:
        for raw in explicit_paths:
            path = raw.expanduser()
            if _is_zarr_path(path):
                explicit_zarrs.append(path)
            else:
                roots.append(path)
        if roots:
            roots = _resolve_root(roots)
    else:
        roots = _resolve_root(args.paths)

    log_roots = roots
    if not log_roots and explicit_zarrs:
        log_roots = [_infer_recording_dir(explicit_zarrs[0]).parent]
    skip_existing = not args.overwrite

    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, log_roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_keypoints_batch_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            roots=[str(root) for root in roots],
            zarr_paths=[str(path) for path in explicit_zarrs],
            file_list=[str(path) for path in (args.file_list or [])],
            recursive=bool(args.recursive),
            apply=bool(args.apply),
            dry_run=not bool(args.apply),
            overwrite=bool(args.overwrite),
            require_background=bool(args.require_background) and not bool(args.no_require_background),
            require_crop=bool(args.require_crop) and not bool(args.no_require_crop),
            require_tuning=bool(args.require_tuning),
            config=args.config,
            method=args.method,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            quiet=bool(args.quiet),
            refine=bool(args.refine),
            refine_only=bool(args.refine_only),
            dask_progress=bool(args.dask_progress),
            json=bool(args.json),
        )

    config = _load_config(args.config)
    method = _resolve_method(config, args.method)
    require_background = bool(args.require_background) and not bool(args.no_require_background)
    require_crop = bool(args.require_crop) and not bool(args.no_require_crop)

    plans: List[KeypointPlan] = []
    if roots:
        plans.extend(
            _build_plans(
                roots,
                args.recursive,
                skip_existing=skip_existing,
                require_crop=require_crop,
                require_background=require_background,
                require_tuning=bool(args.require_tuning),
                refine_only=bool(args.refine_only),
            )
        )
    if explicit_zarrs:
        plans.extend(
            _build_plans_from_zarr(
                explicit_zarrs,
                skip_existing=skip_existing,
                require_crop=require_crop,
                require_background=require_background,
                require_tuning=bool(args.require_tuning),
                refine_only=bool(args.refine_only),
            )
        )

    if plans:
        seen: set[str] = set()
        unique: List[KeypointPlan] = []
        for plan in plans:
            key = str(plan.zarr_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(plan)
        plans = unique

    if not args.apply:
        console = Console() if Console is not None else None
        if console is not None:
            console.rule("[bold yellow]Dry run[/bold yellow]")
            console.print("Add [cyan]--apply[/cyan] to run keypoints.")
        else:
            print("Dry run: add --apply to run keypoints.")
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
                        "keypoints_plan",
                        recording=str(plan.recording_dir),
                        zarr=str(plan.zarr_path),
                        camera_id=plan.camera_id,
                        status=plan.status,
                        reason=plan.reason,
                        keypoints_present=plan.keypoints_present,
                        crop_present=plan.crop_present,
                        background_present=plan.background_present,
                        tuning_present=plan.tuning_present,
                    )
        else:
            _print_plan(plans)
            if logger is not None:
                for plan in plans:
                    logger.log(
                        "keypoints_plan",
                        recording=str(plan.recording_dir),
                        zarr=str(plan.zarr_path),
                        camera_id=plan.camera_id,
                        status=plan.status,
                        reason=plan.reason,
                        keypoints_present=plan.keypoints_present,
                        crop_present=plan.crop_present,
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

    runnable_plans: List[KeypointPlan] = []
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            if args.json:
                print(json.dumps({"status": "missing", "zarr": str(plan.zarr_path)}))
            else:
                print(f"Skipping (missing prerequisites): {plan.zarr_path} ({plan.reason})")
            if logger is not None:
                logger.log(
                    "keypoints_skipped",
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
                print(f"Skipping (keypoints exist): {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "keypoints_skipped",
                    zarr=str(plan.zarr_path),
                    status="skipped",
                    reason="keypoints run already present",
                )
            continue
        runnable_plans.append(plan)

    console = Console() if (Console is not None and not args.json and not args.quiet) else None
    progress = _progress(console if not args.json else None, total=len(runnable_plans))
    quiet = bool(args.quiet or args.json)
    task_label = "Refining keypoints" if args.refine_only else "Running keypoints"

    def _run_plan(plan: KeypointPlan) -> Optional[str]:
        if args.refine_only:
            run_name = _latest_keypoints_run(plan.zarr_path)
            total_rois = _keypoints_total_rois(plan.zarr_path, run_name)
            if total_rois is None or total_rois > 0:
                _run_refine(str(plan.zarr_path), config, run_name, quiet=quiet)
            return run_name

        if method in {"yolo", "yolo_pose"}:
            run_name = _run_yolo(str(plan.zarr_path), config, quiet=quiet)
        else:
            _run_traditional(
                str(plan.zarr_path),
                config,
                scheduler=args.scheduler,
                num_workers=args.num_workers,
                quiet=quiet,
                show_progress=args.dask_progress and not args.quiet and not args.json,
            )
            run_name = _latest_keypoints_run(plan.zarr_path)
        if args.refine:
            total_rois = _keypoints_total_rois(plan.zarr_path, run_name)
            if total_rois is None or total_rois > 0:
                _run_refine(str(plan.zarr_path), config, run_name, quiet=quiet)
        return run_name

    if progress is None:
        for plan in runnable_plans:
            try:
                _run_plan(plan)
                ok += 1
                if args.json:
                    print(json.dumps({"status": "ok", "zarr": str(plan.zarr_path)}))
                if logger is not None:
                    logger.log(
                        "keypoints_ok",
                        zarr=str(plan.zarr_path),
                    )
            except Exception as exc:
                failed += 1
                if args.json:
                    print(json.dumps({"status": "failed", "zarr": str(plan.zarr_path), "error": str(exc)}))
                else:
                    print(f"Keypoints failed for {plan.zarr_path}: {exc}")
                if logger is not None:
                    logger.log(
                        "keypoints_failed",
                        zarr=str(plan.zarr_path),
                        error=str(exc),
                    )
    else:
        with progress:
            task = progress.add_task(task_label, total=len(runnable_plans))
            for plan in runnable_plans:
                try:
                    _run_plan(plan)
                    ok += 1
                    if args.json:
                        print(json.dumps({"status": "ok", "zarr": str(plan.zarr_path)}))
                    if logger is not None:
                        logger.log(
                            "keypoints_ok",
                            zarr=str(plan.zarr_path),
                        )
                except Exception as exc:
                    failed += 1
                    if args.json:
                        print(json.dumps({"status": "failed", "zarr": str(plan.zarr_path), "error": str(exc)}))
                    else:
                        print(f"Keypoints failed for {plan.zarr_path}: {exc}")
                    if logger is not None:
                        logger.log(
                            "keypoints_failed",
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
