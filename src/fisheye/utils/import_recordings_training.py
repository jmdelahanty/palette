#!/usr/bin/env python3
"""
Batch import sampled training Zarrs from the recordings layout.

Defaults:
  - Input: camera video in recording_dir/cams/*.mp4
  - Output Zarr: recording_dir/zarr/<recording_dir>_training.zarr
  - Mode: sampled training import (requires --frame-step)
"""

from __future__ import annotations

import argparse
import json
import math
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
DECODE_BACKEND_PYNVVC_LUMA = "pynvvc-luma"
DECODE_BACKENDS = (DECODE_BACKEND_PYNVVC_LUMA,)


_utc_now = utc_now
JsonLogger = SharedJsonLogger


@dataclass
class ImportPlan:
    recording_dir: Path
    h5_path: Path
    camera_id: Optional[str]
    cam_video: Optional[Path]
    zarr_path: Path
    frame_step: Optional[int]
    status: str
    reason: Optional[str] = None
    source_frame_count: Optional[int] = None
    target_sampled_frames: Optional[int] = None
    estimated_sampled_frames: Optional[int] = None
    frame_count_source: Optional[str] = None
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


def _training_zarr_path(recording_dir: Path) -> Path:
    return recording_dir / "zarr" / f"{recording_dir.name}_training.zarr"


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


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _coerce_positive_int(value: object) -> Optional[int]:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _read_manifest_frame_count(recording_dir: Path) -> Tuple[Optional[int], Optional[str]]:
    manifest = _safe_read_json(recording_dir / "recording_manifest.json")
    if not manifest:
        return None, None

    streams = ((manifest.get("video_streams") or {}).get("streams") or {})
    if isinstance(streams, dict):
        for stream_name in ("full", "camera", "crop"):
            stream = streams.get(stream_name)
            if isinstance(stream, dict):
                frame_count = _coerce_positive_int(stream.get("frame_count"))
                if frame_count is not None:
                    return frame_count, f"recording_manifest.video_streams.{stream_name}.frame_count"

    files = manifest.get("files") or {}
    if isinstance(files, dict):
        for rel_path in files.get("cams") or []:
            rel_text = str(rel_path)
            if not rel_text.endswith("_external_summary.json"):
                continue
            summary = _safe_read_json(recording_dir / rel_text)
            if not summary:
                continue
            for key in ("frames_encoded", "frames_received", "acks_sent"):
                frame_count = _coerce_positive_int(summary.get(key))
                if frame_count is not None:
                    return frame_count, f"{rel_text}:{key}"
            ipc = summary.get("ipc_protocol")
            if isinstance(ipc, dict):
                frame_count = _coerce_positive_int(
                    ipc.get("client_finalize_frame_count") or ipc.get("client_drain_first_frame_count")
                )
                if frame_count is not None:
                    return frame_count, f"{rel_text}:ipc_protocol.client_finalize_frame_count"

    return None, None


def _estimate_sampled_frames(source_frame_count: int, frame_step: int, skip_tail_frames: int) -> int:
    effective_total = max(int(source_frame_count) - max(int(skip_tail_frames), 0), 0)
    if effective_total <= 0:
        return 0
    return ((effective_total - 1) // int(frame_step)) + 1


def _resolve_frame_step(
    *,
    source_frame_count: Optional[int],
    requested_frame_step: Optional[int],
    target_sampled_frames: Optional[int],
    skip_tail_frames: int,
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if target_sampled_frames is not None:
        if source_frame_count is not None:
            effective_total = max(source_frame_count - max(skip_tail_frames, 0), 0)
            if effective_total <= 0:
                return None, 0, "source frame count is not larger than --skip-tail-frames"
            frame_step = max(1, math.ceil(effective_total / target_sampled_frames))
            return frame_step, _estimate_sampled_frames(source_frame_count, frame_step, skip_tail_frames), None
        if requested_frame_step is None:
            return None, None, "--target-sampled-frames requires frame-count metadata or --frame-step fallback"

    frame_step = requested_frame_step
    if frame_step is None:
        frame_step = 100
    estimated = (
        _estimate_sampled_frames(source_frame_count, frame_step, skip_tail_frames)
        if source_frame_count is not None
        else None
    )
    return frame_step, estimated, None


def _build_plans(
    root: Path,
    recursive: bool,
    skip_existing: bool,
    check_stimulus: bool,
    requested_frame_step: Optional[int],
    target_sampled_frames: Optional[int],
    skip_tail_frames: int,
    path_contains: Optional[str] = None,
    limit: Optional[int] = None,
    require_source_frame_count: bool = False,
) -> List[ImportPlan]:
    plans: List[ImportPlan] = []
    for h5_path in _find_h5_files(root, recursive):
        if limit is not None and len(plans) >= limit:
            break
        if h5_path.parent.name != "raw":
            continue
        recording_dir = h5_path.parent.parent
        if path_contains and path_contains not in str(recording_dir):
            continue
        zarr_path = _training_zarr_path(recording_dir)
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
                    frame_step=None,
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
        source_frame_count, frame_count_source = _read_manifest_frame_count(recording_dir)
        frame_step, estimated_sampled, frame_step_reason = _resolve_frame_step(
            source_frame_count=source_frame_count,
            requested_frame_step=requested_frame_step,
            target_sampled_frames=target_sampled_frames,
            skip_tail_frames=skip_tail_frames,
        )
        if frame_step is None:
            status = "missing"
            reason = frame_step_reason
        elif require_source_frame_count and source_frame_count is None:
            status = "missing"
            reason = "source frame count is required for PyNvVC sequential sampled import"
        elif skip_existing and zarr_path.exists():
            status = "skipped"
            reason = "zarr already exists"
        stimulus_present: Optional[bool] = None
        existing_frame_step: Optional[int] = None
        frame_step_mismatch: Optional[bool] = None
        if zarr_path.exists():
            existing_frame_step = _read_existing_frame_step(zarr_path)
            if existing_frame_step is not None and frame_step is not None:
                frame_step_mismatch = existing_frame_step != frame_step
        if check_stimulus and zarr_path.exists():
            stimulus_present = _stimulus_runs_present(zarr_path)
        plans.append(
            ImportPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                camera_id=camera_id,
                cam_video=cam_video,
                zarr_path=zarr_path,
                frame_step=frame_step,
                status=status,
                reason=reason,
                source_frame_count=source_frame_count,
                target_sampled_frames=target_sampled_frames,
                estimated_sampled_frames=estimated_sampled,
                frame_count_source=frame_count_source,
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
        print(f"  frame_step: {plan.frame_step or 'unresolved'}")
        if plan.source_frame_count is not None:
            print(
                f"  source_frame_count: {plan.source_frame_count} "
                f"({plan.frame_count_source or 'unknown source'})"
            )
        if plan.estimated_sampled_frames is not None:
            print(f"  estimated_sampled_frames: {plan.estimated_sampled_frames}")
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
    table.add_column("Frames", justify="right")
    table.add_column("Sample", justify="right")
    table.add_column("Mismatch", justify="center")

    counts = {"ok": 0, "skipped": 0, "missing": 0}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
        cam_name = plan.cam_video.name if plan.cam_video else "MISSING"
        zarr_name = plan.zarr_path.name
        step = "-" if plan.frame_step is None else str(plan.frame_step)
        frames = "-" if plan.source_frame_count is None else str(plan.source_frame_count)
        sample = "-" if plan.estimated_sampled_frames is None else str(plan.estimated_sampled_frames)
        mismatch = "!" if plan.frame_step_mismatch else ""
        table.add_row(
            plan.recording_dir.name,
            plan.status,
            plan.camera_id or "unknown",
            cam_name,
            zarr_name,
            step,
            frames,
            sample,
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
    overwrite: bool,
    skip_tail_frames: int,
    decode_backend: str,
    gpu_id: int,
    scratch_root: Path,
) -> Tuple[bool, int]:
    if plan.frame_step is None:
        return False, 2
    plan.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    if decode_backend == DECODE_BACKEND_PYNVVC_LUMA:
        if plan.source_frame_count is None:
            return False, 2
        if overwrite:
            raise ValueError("Atomic sampled-training publication refuses overwrite.")
        if not plan.camera_id:
            raise ValueError("Atomic sampled-training publication requires camera_id.")
        cmd = [
            sys.executable,
            "-m",
            "fisheye.utils.publish_sampled_training_base",
            "--destination",
            str(plan.zarr_path),
            "--scratch-root",
            str(scratch_root),
            "--video-path",
            str(plan.cam_video),
            "--source-frame-count",
            str(plan.source_frame_count),
            "--frame-step",
            str(plan.frame_step),
            "--config",
            str(config_path),
            "--camera-id",
            str(plan.camera_id),
            "--recording-dir",
            str(plan.recording_dir),
            "--h5-path",
            str(plan.h5_path),
            "--gpu-id",
            str(int(gpu_id)),
        ]
    else:
        raise ValueError(f"Unsupported decode backend: {decode_backend}")
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


def _run_acquisition_crop_video_append(
    plan: ImportPlan,
    *,
    run_name_prefix: str,
    overwrite_run: bool,
    gpu_id: int,
) -> Tuple[bool, int]:
    run_name = f"{run_name_prefix}_{plan.recording_dir.name}"
    cmd = [
        sys.executable,
        "-m",
        "fisheye.utils.append_acquisition_crop_video_training",
        str(plan.zarr_path),
        "--recording-dir",
        str(plan.recording_dir),
        "--run-name",
        run_name,
        "--gpu-id",
        str(int(gpu_id)),
        "--apply",
    ]
    if overwrite_run:
        cmd.append("--overwrite-run")
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
        help=(
            "Import every Nth frame (training_data mode). If omitted with no "
            "--target-sampled-frames, defaults to 100. If used with "
            "--target-sampled-frames, this is only a fallback when frame-count "
            "metadata is unavailable."
        ),
    )
    parser.add_argument(
        "--target-sampled-frames",
        type=int,
        help=(
            "Target sampled frames per recording. The wrapper computes "
            "frame_step=ceil((source_frame_count-skip_tail_frames)/target)."
        ),
    )
    parser.add_argument(
        "--path-contains",
        type=str,
        help="Only include recordings whose directory path contains this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of planned recordings after filtering. Useful for one-recording smokes.",
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
        help="Retired: publish a new versioned training Zarr instead.",
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
    parser.add_argument(
        "--decode-backend",
        choices=DECODE_BACKENDS,
        default=DECODE_BACKEND_PYNVVC_LUMA,
        help=(
            "Decode backend for sampled training imports. PyNvVC luma is the sole "
            "supported writer contract."
        ),
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id for PyNvVC sampled training imports.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help=(
            "Bounded node-local scratch directory required when applying. "
            "All sampled training bases are published atomically."
        ),
    )
    parser.add_argument(
        "--include-acquisition-crop-video",
        action="store_true",
        help=(
            "After importing sampled full-frame training images, append sampled "
            "acquisition crop-video frames into crop_runs/<run> in the same training zarr."
        ),
    )
    parser.add_argument(
        "--acquisition-crop-run-prefix",
        default="crop_acquisition_crop_video_training",
        help="Run-name prefix for --include-acquisition-crop-video.",
    )
    parser.add_argument(
        "--overwrite-acquisition-crop-run",
        action="store_true",
        help="Overwrite an existing acquisition crop-video crop run with the generated run name.",
    )

    args = parser.parse_args(argv)

    if not args.apply:
        args.dry_run = True

    root = _resolve_root(args.recordings_root)
    if not root.exists():
        print(f"Recordings root not found: {root}")
        return 1

    if args.frame_step is not None and args.frame_step < 1:
        print(f"--frame-step must be >= 1 (got {args.frame_step})")
        return 1
    if args.target_sampled_frames is not None and args.target_sampled_frames < 1:
        print(f"--target-sampled-frames must be >= 1 (got {args.target_sampled_frames})")
        return 1
    if args.limit is not None and args.limit < 1:
        print(f"--limit must be >= 1 (got {args.limit})")
        return 1
    if args.skip_tail_frames < 0:
        print(f"--skip-tail-frames must be >= 0 (got {args.skip_tail_frames})")
        return 1
    if args.gpu_id < 0:
        print(f"--gpu-id must be >= 0 (got {args.gpu_id})")
        return 1
    if args.overwrite:
        print(
            "--overwrite is retired for sampled training publication; publish a new "
            "versioned artifact instead."
        )
        return 1
    if args.apply and args.scratch_root is None:
        print("Sampled training publication with --apply requires --scratch-root")
        return 1
    skip_existing = not args.overwrite and not args.no_skip_existing
    requested_frame_step = args.frame_step
    if requested_frame_step is None and args.target_sampled_frames is None:
        requested_frame_step = 100

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
            frame_step=int(requested_frame_step) if requested_frame_step is not None else None,
            target_sampled_frames=(
                int(args.target_sampled_frames) if args.target_sampled_frames is not None else None
            ),
            skip_tail_frames=int(args.skip_tail_frames),
            config=str(args.config),
            dry_run=bool(args.dry_run),
            apply=bool(args.apply),
            overwrite=bool(args.overwrite),
            skip_existing=bool(skip_existing),
            register=bool(args.register),
            import_stimulus=bool(args.import_stimulus),
            path_contains=args.path_contains,
            limit=args.limit,
            decode_backend=args.decode_backend,
            gpu_id=int(args.gpu_id),
            scratch_root=str(args.scratch_root) if args.scratch_root is not None else None,
            include_acquisition_crop_video=bool(args.include_acquisition_crop_video),
            acquisition_crop_run_prefix=str(args.acquisition_crop_run_prefix),
        )

    if args.apply:
        _gpu_preflight(logger)
    plans = _build_plans(
        root,
        args.recursive,
        skip_existing=skip_existing,
        check_stimulus=args.import_stimulus,
        requested_frame_step=requested_frame_step,
        target_sampled_frames=args.target_sampled_frames,
        skip_tail_frames=args.skip_tail_frames,
        path_contains=args.path_contains,
        limit=args.limit,
        require_source_frame_count=True,
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
                    frame_step=plan.frame_step,
                    source_frame_count=plan.source_frame_count,
                    target_sampled_frames=plan.target_sampled_frames,
                    estimated_sampled_frames=plan.estimated_sampled_frames,
                    frame_count_source=plan.frame_count_source,
                    existing_frame_step=plan.existing_frame_step,
                    frame_step_mismatch=plan.frame_step_mismatch,
                    decode_backend=args.decode_backend,
                    gpu_id=int(args.gpu_id),
                    include_acquisition_crop_video=bool(args.include_acquisition_crop_video),
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
                    f"differs from planned {plan.frame_step} for {plan.zarr_path}"
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
                frame_step=plan.frame_step,
                source_frame_count=plan.source_frame_count,
                target_sampled_frames=plan.target_sampled_frames,
                estimated_sampled_frames=plan.estimated_sampled_frames,
                frame_count_source=plan.frame_count_source,
                decode_backend=args.decode_backend,
                gpu_id=int(args.gpu_id),
            )
        success, returncode = _run_import(
            plan,
            config_path=args.config,
            overwrite=args.overwrite,
            skip_tail_frames=args.skip_tail_frames,
            decode_backend=args.decode_backend,
            gpu_id=int(args.gpu_id),
            scratch_root=args.scratch_root,
        )
        if success:
            if args.include_acquisition_crop_video:
                if logger is not None:
                    logger.log(
                        "acquisition_crop_video_append_start",
                        recording_dir=str(plan.recording_dir),
                        zarr_path=str(plan.zarr_path),
                        run_name_prefix=str(args.acquisition_crop_run_prefix),
                    )
                crop_ok, crop_returncode = _run_acquisition_crop_video_append(
                    plan,
                    run_name_prefix=str(args.acquisition_crop_run_prefix),
                    overwrite_run=bool(args.overwrite_acquisition_crop_run),
                    gpu_id=int(args.gpu_id),
                )
                if not crop_ok:
                    failed += 1
                    print(f"Acquisition crop-video append failed for {plan.zarr_path}")
                    if logger is not None:
                        logger.log(
                            "acquisition_crop_video_append_failed",
                            recording_dir=str(plan.recording_dir),
                            zarr_path=str(plan.zarr_path),
                            returncode=crop_returncode,
                        )
                    continue
                if logger is not None:
                    logger.log(
                        "acquisition_crop_video_append_success",
                        recording_dir=str(plan.recording_dir),
                        zarr_path=str(plan.zarr_path),
                        returncode=crop_returncode,
                    )
            ok += 1
            if logger is not None:
                logger.log(
                    "import_success",
                    recording_dir=str(plan.recording_dir),
                    zarr_path=str(plan.zarr_path),
                    returncode=returncode,
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
