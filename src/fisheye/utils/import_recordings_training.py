#!/usr/bin/env python3
"""
Batch import sampled training Zarrs from the recordings layout.

Defaults:
  - Input: camera video in recording_dir/cams/*.mp4
  - Output Zarr: recording_dir/zarr/<h5_stem>.zarr
  - Mode: sampled training import (requires --frame-step)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import zarr

from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr as _normalize_attr

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - rich is optional here
    Console = None
    Table = None

from fisheye.registry.db import Registry, RegistryPaths


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


_utc_now = utc_now
JsonLogger = SharedJsonLogger


@dataclass
class ImportPlan:
    recording_dir: Path
    h5_path: Path
    camera_id: Optional[str]
    cam_video: Optional[Path]
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    stimulus_present: Optional[bool] = None
    existing_frame_step: Optional[int] = None
    frame_step_mismatch: Optional[bool] = None


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


def _read_h5_meta(h5_path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with h5py.File(h5_path, "r") as h5:
        root = h5.attrs
        keys = (
            "session_uuid",
            "session_start_iso8601_utc",
            "arena_id",
            "camera_id",
            "ipc_source_name",
            "protocol_name_from_definition",
        )
        for key in keys:
            if key in root:
                value = _normalize_attr(root.get(key))
                if value:
                    meta[key] = value
        if "camera_id" not in meta:
            derived = _derive_camera_id(meta.get("ipc_source_name"))
            if derived:
                meta["camera_id"] = derived
                meta["camera_id_source"] = "ipc_source_name"
    return meta


def _find_h5_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(root.rglob("raw/*.h5"))
    return sorted(root.glob("*/raw/*.h5"))


def _select_cam_video(recording_dir: Path, camera_id: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    cams_dir = recording_dir / "cams"
    if not cams_dir.exists():
        return None, "missing cams/ directory"
    mp4s = sorted(cams_dir.glob("*.mp4"))
    if not mp4s:
        return None, "no .mp4 files in cams/"
    if len(mp4s) == 1:
        return mp4s[0], None
    if camera_id:
        matches = [path for path in mp4s if f"Cam{camera_id}" in path.stem]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"multiple cam videos matched camera_id {camera_id}"
    return None, "multiple cam videos and no unique match"


def _build_plans(
    root: Path,
    recursive: bool,
    skip_existing: bool,
    check_stimulus: bool,
    requested_frame_step: int,
) -> List[ImportPlan]:
    plans: List[ImportPlan] = []
    for h5_path in _find_h5_files(root, recursive):
        if h5_path.parent.name != "raw":
            continue
        recording_dir = h5_path.parent.parent
        zarr_path = recording_dir / "zarr" / f"{h5_path.stem}.zarr"
        try:
            meta = _read_h5_meta(h5_path)
        except Exception as exc:
            plans.append(
                ImportPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    camera_id=None,
                    cam_video=None,
                    zarr_path=zarr_path,
                    status="missing",
                    reason=f"failed to read H5: {exc}",
                )
            )
            continue
        camera_id = meta.get("camera_id")
        cam_video, reason = _select_cam_video(recording_dir, camera_id)
        status = "ok"
        if cam_video is None:
            status = "missing"
        elif skip_existing and zarr_path.exists():
            status = "skipped"
            reason = "zarr already exists"
        stimulus_present: Optional[bool] = None
        existing_frame_step: Optional[int] = None
        frame_step_mismatch: Optional[bool] = None
        if zarr_path.exists():
            existing_frame_step = _read_existing_frame_step(zarr_path)
            if existing_frame_step is not None:
                frame_step_mismatch = existing_frame_step != requested_frame_step
        if check_stimulus and zarr_path.exists():
            stimulus_present = _stimulus_runs_present(zarr_path)
        plans.append(
            ImportPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                camera_id=camera_id,
                cam_video=cam_video,
                zarr_path=zarr_path,
                status=status,
                reason=reason,
                stimulus_present=stimulus_present,
                existing_frame_step=existing_frame_step,
                frame_step_mismatch=frame_step_mismatch,
            )
        )
    return plans


def _print_plan(plans: List[ImportPlan]) -> None:
    if not plans:
        print("No recordings found.")
        return
    counts = {"ok": 0, "skipped": 0, "missing": 0}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
        print(f"Recording: {plan.recording_dir.name}")
        print(f"  h5: {plan.h5_path}")
        print(f"  camera_id: {plan.camera_id or 'unknown'}")
        print(f"  cam: {plan.cam_video or 'MISSING'}")
        print(f"  zarr: {plan.zarr_path}")
        if plan.existing_frame_step is not None:
            mismatch = " (mismatch)" if plan.frame_step_mismatch else ""
            print(f"  existing_frame_step: {plan.existing_frame_step}{mismatch}")
        if plan.stimulus_present is not None:
            stimulus_label = "present" if plan.stimulus_present else "missing"
            print(f"  stimulus_runs: {stimulus_label}")
        print(f"  status: {plan.status}" + (f" ({plan.reason})" if plan.reason else ""))
        print("")
    print("Summary:")
    print(f"  ok: {counts.get('ok', 0)}")
    print(f"  skipped: {counts.get('skipped', 0)}")
    print(f"  missing: {counts.get('missing', 0)}")


def _print_plan_rich(plans: List[ImportPlan]) -> None:
    if Console is None or Table is None:
        _print_plan(plans)
        return
    if not plans:
        Console().print("No recordings found.")
        return

    console = Console()
    table = Table(title="Planned training imports", show_lines=False)
    table.add_column("Recording", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Camera", justify="right")
    table.add_column("Cam video")
    table.add_column("Zarr")
    table.add_column("Step", justify="right")
    table.add_column("Mismatch", justify="center")

    counts = {"ok": 0, "skipped": 0, "missing": 0}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
        cam_name = plan.cam_video.name if plan.cam_video else "MISSING"
        zarr_name = plan.zarr_path.name
        step = "-" if plan.existing_frame_step is None else str(plan.existing_frame_step)
        mismatch = "!" if plan.frame_step_mismatch else ""
        table.add_row(
            plan.recording_dir.name,
            plan.status,
            plan.camera_id or "unknown",
            cam_name,
            zarr_name,
            step,
            mismatch,
        )

    console.print(table)
    console.print(
        f"Summary: ok={counts.get('ok', 0)} skipped={counts.get('skipped', 0)} missing={counts.get('missing', 0)}"
    )


def _run_import(
    plan: ImportPlan,
    *,
    config_path: Path,
    frame_step: int,
    overwrite: bool,
    skip_tail_frames: int,
) -> Tuple[bool, int]:
    plan.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "fisheye.capture.import_video",
        str(plan.cam_video),
        "--config",
        str(config_path),
        "--training-data",
        "--frame-step",
        str(frame_step),
        "--zarr-path",
        str(plan.zarr_path),
    ]
    if skip_tail_frames:
        cmd.extend(["--skip-tail-frames", str(skip_tail_frames)])
    if overwrite:
        cmd.append("--overwrite")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode


def _stimulus_runs_present(zarr_path: Path) -> bool:
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception:
        return False
    analysis = root.get("analysis")
    if analysis is None:
        return False
    stim = analysis.get("stimulus_runs")
    if stim is None:
        return False
    return len(list(stim.group_keys())) > 0


def _read_existing_frame_step(zarr_path: Path) -> Optional[int]:
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception:
        return None
    raw = root.get("raw_video")
    if raw is None:
        return None
    value = raw.attrs.get("frame_step")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_stimulus_import(
    plan: ImportPlan,
    *,
    run_name: Optional[str],
    overwrite: bool,
    quiet: bool,
) -> Tuple[bool, int]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.import_stimulus_to_zarr",
        str(plan.h5_path),
        str(plan.zarr_path),
    ]
    if run_name:
        cmd.extend(["--run-name", run_name])
    if overwrite:
        cmd.append("--overwrite")
    if quiet:
        cmd.append("--quiet")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode


def _resolve_root(arg_root: Optional[Path]) -> Path:
    if arg_root is not None:
        return arg_root
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return Path(env_root)
    return DEFAULT_RECORDINGS_ROOT


def _resolve_log_dir(arg_log_dir: Optional[Path], recordings_root: Path) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "import_recordings_training"
    return recordings_root / "logs" / "import_recordings_training"


_run_id = make_run_id


def _gpu_preflight(logger: Optional[JsonLogger]) -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        message = "GPU preflight failed: nvidia-smi not found (no NVIDIA driver available)."
        print(message)
        if logger is not None:
            logger.log("preflight_failed", reason=message)
            logger.log("run_end", ok=0, failed=0, skipped=0, missing=0)
            logger.close()
        raise SystemExit(2)

    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        if "Driver/library version mismatch" in details or "Failed to initialize NVML" in details:
            message = (
                "GPU preflight failed: NVML driver/library version mismatch. "
                "Reboot to reload the NVIDIA kernel module or run on a node with a working driver."
            )
        else:
            message = f"GPU preflight failed: nvidia-smi returned {result.returncode}. {details}"
        print(message)
        if logger is not None:
            logger.log("preflight_failed", reason=message, details=details)
            logger.log("run_end", ok=0, failed=0, skipped=0, missing=0)
            logger.close()
        raise SystemExit(2)

    if logger is not None:
        logger.log("preflight_ok")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch import sampled training Zarrs from recordings.",
    )
    parser.add_argument(
        "recordings_root",
        nargs="?",
        type=Path,
        help="Root recordings directory (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fisheye/import_local.yaml"),
        help="Import config YAML to use.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=100,
        help="Import every Nth frame (training_data mode).",
    )
    parser.add_argument(
        "--skip-tail-frames",
        type=int,
        default=200,
        help="Skip the last N frames to avoid EOF/GOP issues during decoding.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for recordings under the root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned imports without running them (default behavior).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run imports.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Zarrs (disables skipping).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Attempt imports even if Zarr exists (unless --overwrite is set).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register created Zarrs in the registry.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path.",
    )
    parser.add_argument(
        "--import-stimulus",
        action="store_true",
        help="After import, mirror stimulus H5 into analysis/stimulus_runs (skips if already present).",
    )
    parser.add_argument(
        "--stimulus-always",
        action="store_true",
        help="Always run stimulus import even if stimulus runs already exist.",
    )
    parser.add_argument(
        "--stimulus-run-name",
        type=str,
        help="Optional stimulus run name (defaults to timestamped name).",
    )
    parser.add_argument(
        "--stimulus-overwrite",
        action="store_true",
        help="Overwrite an existing stimulus run name.",
    )
    parser.add_argument(
        "--stimulus-quiet",
        action="store_true",
        help="Suppress verbose stimulus import output.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/import_recordings_training or <recordings_root>/logs/import_recordings_training).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable JSONL logging.",
    )
    parser.add_argument(
        "--rich",
        action="store_true",
        help="Use rich-formatted console output for dry-run summaries.",
    )

    args = parser.parse_args(argv)

    if not args.apply:
        args.dry_run = True

    root = _resolve_root(args.recordings_root)
    if not root.exists():
        print(f"Recordings root not found: {root}")
        return 1

    if args.frame_step < 1:
        print(f"--frame-step must be >= 1 (got {args.frame_step})")
        return 1
    if args.skip_tail_frames < 0:
        print(f"--skip-tail-frames must be >= 0 (got {args.skip_tail_frames})")
        return 1

    skip_existing = not args.overwrite and not args.no_skip_existing

    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, root)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"import_recordings_training_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            recordings_root=str(root),
            recursive=bool(args.recursive),
            frame_step=int(args.frame_step),
            skip_tail_frames=int(args.skip_tail_frames),
            config=str(args.config),
            dry_run=bool(args.dry_run),
            apply=bool(args.apply),
            overwrite=bool(args.overwrite),
            skip_existing=bool(skip_existing),
            register=bool(args.register),
            import_stimulus=bool(args.import_stimulus),
        )

    if args.apply:
        _gpu_preflight(logger)
    plans = _build_plans(
        root,
        args.recursive,
        skip_existing=skip_existing,
        check_stimulus=args.import_stimulus,
        requested_frame_step=args.frame_step,
    )

    if args.dry_run:
        print("Planned training imports (dry-run):")
        if logger is not None:
            for plan in plans:
                logger.log(
                    "recording_plan",
                    recording_dir=str(plan.recording_dir),
                    h5_path=str(plan.h5_path),
                    camera_id=plan.camera_id,
                    cam_video=str(plan.cam_video) if plan.cam_video else None,
                    zarr_path=str(plan.zarr_path),
                    status=plan.status,
                    reason=plan.reason,
                    existing_frame_step=plan.existing_frame_step,
                    frame_step_mismatch=plan.frame_step_mismatch,
                )
        if args.rich:
            _print_plan_rich(plans)
        else:
            _print_plan(plans)
        if logger is not None:
            logger.log("run_end", ok=0, failed=0, skipped=0, missing=0)
            logger.close()
        return 0

    registry: Optional[Registry] = None
    if args.register:
        registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
        registry = Registry(registry_path)
        print(f"Registry: {registry_path}")

    ok = 0
    failed = 0
    skipped = 0
    missing = 0
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            print(f"Skipping (missing): {plan.recording_dir.name} ({plan.reason})")
            if logger is not None:
                logger.log(
                    "recording_skipped",
                    recording_dir=str(plan.recording_dir),
                    reason=plan.reason,
                    status="missing",
                )
            continue
        if plan.status == "skipped" and not args.overwrite and not args.no_skip_existing:
            skipped += 1
            if plan.frame_step_mismatch:
                print(
                    f"Warning: existing frame_step={plan.existing_frame_step} "
                    f"differs from requested {args.frame_step} for {plan.zarr_path}"
                )
            print(f"Skipping (exists): {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "recording_skipped",
                    recording_dir=str(plan.recording_dir),
                    reason=plan.reason or "zarr exists",
                    status="skipped",
                    existing_frame_step=plan.existing_frame_step,
                    frame_step_mismatch=plan.frame_step_mismatch,
                )
            continue
        if logger is not None:
            logger.log(
                "import_start",
                recording_dir=str(plan.recording_dir),
                h5_path=str(plan.h5_path),
                cam_video=str(plan.cam_video),
                zarr_path=str(plan.zarr_path),
            )
        success, returncode = _run_import(
            plan,
            config_path=args.config,
            frame_step=args.frame_step,
            overwrite=args.overwrite,
            skip_tail_frames=args.skip_tail_frames,
        )
        if success:
            ok += 1
            if logger is not None:
                logger.log(
                    "import_success",
                    recording_dir=str(plan.recording_dir),
                    zarr_path=str(plan.zarr_path),
                    returncode=returncode,
                )
            if registry is not None:
                try:
                    dataset_id = registry.scan_zarr(plan.zarr_path)
                    if dataset_id:
                        print(f"Registered dataset: {dataset_id}")
                    if logger is not None:
                        logger.log(
                            "registry_register",
                            recording_dir=str(plan.recording_dir),
                            zarr_path=str(plan.zarr_path),
                            dataset_id=dataset_id,
                        )
                except Exception as exc:
                    print(f"Registry warning: {exc}")
                    if logger is not None:
                        logger.log(
                            "registry_warning",
                            recording_dir=str(plan.recording_dir),
                            zarr_path=str(plan.zarr_path),
                            error=str(exc),
                        )
            if args.import_stimulus:
                stim_present = _stimulus_runs_present(plan.zarr_path)
                if stim_present and not args.stimulus_always:
                    print("Stimulus runs already present; skipping stimulus import.")
                    if logger is not None:
                        logger.log(
                            "stimulus_skipped",
                            recording_dir=str(plan.recording_dir),
                            zarr_path=str(plan.zarr_path),
                            reason="stimulus_runs already present",
                        )
                else:
                    if logger is not None:
                        logger.log(
                            "stimulus_start",
                            recording_dir=str(plan.recording_dir),
                            zarr_path=str(plan.zarr_path),
                            h5_path=str(plan.h5_path),
                        )
                    stim_ok, stim_returncode = _run_stimulus_import(
                        plan,
                        run_name=args.stimulus_run_name,
                        overwrite=args.stimulus_overwrite,
                        quiet=args.stimulus_quiet,
                    )
                    if not stim_ok:
                        print(f"Stimulus import failed for {plan.zarr_path}")
                        if logger is not None:
                            logger.log(
                                "stimulus_failed",
                                recording_dir=str(plan.recording_dir),
                                zarr_path=str(plan.zarr_path),
                                returncode=stim_returncode,
                            )
                    elif logger is not None:
                        logger.log(
                            "stimulus_success",
                            recording_dir=str(plan.recording_dir),
                            zarr_path=str(plan.zarr_path),
                            returncode=stim_returncode,
                        )
        else:
            failed += 1
            print(f"FAILED: {plan.recording_dir.name}")
            if logger is not None:
                logger.log(
                    "import_failed",
                    recording_dir=str(plan.recording_dir),
                    zarr_path=str(plan.zarr_path),
                    returncode=returncode,
                )

    if registry is not None:
        registry.close()

    print("Summary:")
    print(f"  ok: {ok}")
    print(f"  failed: {failed}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    if logger is not None:
        logger.log(
            "run_end",
            ok=ok,
            failed=failed,
            skipped=skipped,
            missing=missing,
        )
        logger.close()
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
