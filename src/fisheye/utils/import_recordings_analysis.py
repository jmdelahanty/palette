#!/usr/bin/env python3
"""Batch-create analysis Zarrs from recordings without training imports.

Workflow per recording:
1) Import/create analysis archive + metadata/stimulus
2) Run YOLO detection (explicit model or registry-resolved)
3) Optionally run refine_detect
4) Optionally register/rescan into the registry

Default mode is dry-run. Use --apply to execute.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils.import_recording_analysis import (
    RecordingImportOptions,
    RecordingAnalysisPlan as SingleRecordingPlan,
    stimulus_runs_present,
)
from fisheye.utils.run_recording_analysis_pipeline import (
    RecordingPipelineOptions,
    process_recording_analysis_pipeline,
)


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


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


@dataclass
class AnalysisPlan:
    recording_dir: Path
    h5_path: Path
    camera_id: Optional[str]
    cam_video: Optional[Path]
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    stimulus_present: Optional[bool] = None


def _normalize_attr(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    text = _normalize_attr(ipc_source_name)
    if not text:
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
        for key in (
            "session_uuid",
            "session_start_iso8601_utc",
            "arena_id",
            "camera_id",
            "ipc_source_name",
            "protocol_name_from_definition",
        ):
            if key in root:
                value = _normalize_attr(root.get(key))
                if value:
                    meta[key] = value
        if "camera_id" not in meta:
            derived = _derive_camera_id(meta.get("ipc_source_name"))
            if derived:
                meta["camera_id"] = derived
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
    # Safety-first behavior: this command is intentionally single-camera today.
    # Multi-camera recordings (single H5 + multiple cam videos) need a dedicated
    # schema/run layout and should not silently collapse to one selected camera.
    return None, "multiple camera videos in recording; multi-camera analysis import is not yet supported by this command"


def _build_plans(
    root: Path,
    recursive: bool,
    skip_existing: bool,
    check_stimulus: bool,
) -> List[AnalysisPlan]:
    # Group by recording so we can detect multi-H5 (future multi-camera) layouts.
    # Current workflow supports one H5/camera stream per recording directory.
    h5_by_recording: Dict[Path, List[Path]] = {}
    for h5_path in _find_h5_files(root, recursive):
        if h5_path.parent.name != "raw":
            continue
        recording_dir = h5_path.parent.parent
        h5_by_recording.setdefault(recording_dir, []).append(h5_path)

    plans: List[AnalysisPlan] = []
    for recording_dir in sorted(h5_by_recording):
        h5_paths = sorted(h5_by_recording[recording_dir])
        if len(h5_paths) > 1:
            reason = (
                f"multiple raw H5 files ({len(h5_paths)}) in recording; "
                "multi-camera analysis import is not yet supported by this command"
            )
            for h5_path in h5_paths:
                zarr_path = recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"
                plans.append(
                    AnalysisPlan(
                        recording_dir=recording_dir,
                        h5_path=h5_path,
                        camera_id=None,
                        cam_video=None,
                        zarr_path=zarr_path,
                        status="missing",
                        reason=reason,
                    )
                )
            continue

        h5_path = h5_paths[0]
        zarr_dir = recording_dir / "zarr"
        zarr_path = zarr_dir / f"{recording_dir.name}_analysis.zarr"
        try:
            meta = _read_h5_meta(h5_path)
        except Exception as exc:
            plans.append(
                AnalysisPlan(
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
            reason = "analysis zarr already exists"

        stimulus_present: Optional[bool] = None
        if check_stimulus and zarr_path.exists():
            stimulus_present = stimulus_runs_present(zarr_path)

        plans.append(
            AnalysisPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                camera_id=camera_id,
                cam_video=cam_video,
                zarr_path=zarr_path,
                status=status,
                reason=reason,
                stimulus_present=stimulus_present,
            )
        )
    return plans


def _print_plan(plans: List[AnalysisPlan]) -> None:
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
        print(f"  analysis_zarr: {plan.zarr_path}")
        if plan.stimulus_present is not None:
            label = "present" if plan.stimulus_present else "missing"
            print(f"  stimulus_runs: {label}")
        print(f"  status: {plan.status}" + (f" ({plan.reason})" if plan.reason else ""))
        print("")
    print("Summary:")
    print(f"  ok: {counts.get('ok', 0)}")
    print(f"  skipped: {counts.get('skipped', 0)}")
    print(f"  missing: {counts.get('missing', 0)}")


def _resolve_root(arg_root: Optional[Path]) -> Path:
    if arg_root is not None:
        return arg_root
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return Path(env_root)
    return DEFAULT_RECORDINGS_ROOT


def _resolve_log_dir(arg_log_dir: Optional[Path], root: Path) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "import_recordings_analysis"
    return root / "logs" / "import_recordings_analysis"


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{os.getpid()}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create analysis Zarrs from recordings using YOLO detect + stimulus import.",
    )
    parser.add_argument(
        "recordings_root",
        nargs="?",
        type=Path,
        help="Root recordings directory (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for recordings.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without running them.")
    parser.add_argument("--apply", action="store_true", help="Run detection/import/refine steps.")
    parser.add_argument("--overwrite", action="store_true", help="Do not skip when analysis zarr already exists.")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Attempt runs even if analysis zarr already exists (unless --overwrite is set).",
    )

    parser.add_argument("--model", type=Path, help="YOLO model path (.pt).")
    parser.add_argument(
        "--model-source",
        choices=("explicit", "registry"),
        default="explicit",
        help="Model resolution source: explicit path/config or per-recording registry resolver.",
    )
    parser.add_argument("--detect-config", type=Path, help="YOLO detect config YAML.")
    parser.add_argument("--conf", type=float, help="YOLO confidence threshold override.")
    parser.add_argument("--iou", type=float, help="YOLO IoU threshold override.")
    parser.add_argument("--max-det", type=int, help="YOLO max detections per frame override.")
    parser.add_argument("--batch-size", type=int, help="YOLO batch size override.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for YOLO detect.")
    parser.add_argument("--set-id", type=str, help="Optional detect set filter when --model-source=registry.")
    parser.add_argument(
        "--require-unique",
        action="store_true",
        help="Fail detect step when top registry candidates are score-tied.",
    )
    parser.add_argument(
        "--include-non-success",
        action="store_true",
        help="Allow non-success training runs as candidates for registry model resolution.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of model candidates to persist in detect run provenance (registry mode).",
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
    parser.set_defaults(import_stimulus=True)
    parser.add_argument("--stimulus-always", action="store_true", help="Run stimulus import even when runs already exist.")
    parser.add_argument("--stimulus-run-name", type=str, help="Optional stimulus run name.")
    parser.add_argument("--stimulus-overwrite", action="store_true", help="Overwrite existing stimulus run name.")
    parser.add_argument("--stimulus-quiet", action="store_true", help="Suppress verbose stimulus import output.")

    parser.add_argument(
        "--refine-detect",
        dest="refine_detect",
        action="store_true",
        help="Run refine_detect after YOLO detect (default).",
    )
    parser.add_argument(
        "--no-refine-detect",
        dest="refine_detect",
        action="store_false",
        help="Skip refine_detect.",
    )
    parser.set_defaults(refine_detect=True)
    parser.add_argument(
        "--refine-config",
        type=Path,
        default=Path("configs/fisheye/default.yaml"),
        help="Config passed to refine_detect.",
    )
    parser.add_argument("--refine-max-gap", type=int, help="Optional max-gap override for refine_detect.")

    parser.add_argument("--register", action="store_true", help="Rescan resulting analysis zarr into registry.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")

    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/import_recordings_analysis or <recordings_root>/logs/import_recordings_analysis).",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging.")

    args = parser.parse_args(argv)

    if not args.apply:
        args.dry_run = True

    root = _resolve_root(args.recordings_root)
    if not root.exists():
        print(f"Recordings root not found: {root}")
        return 1

    skip_existing = not args.overwrite and not args.no_skip_existing

    registry_path = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    if args.model_source == "registry" and not registry_path.exists():
        print(f"Registry not found for --model-source=registry: {registry_path}")
        return 1

    logger: Optional[JsonLogger] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, root)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"import_recordings_analysis_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            recordings_root=str(root),
            recursive=bool(args.recursive),
            dry_run=bool(args.dry_run),
            apply=bool(args.apply),
            skip_existing=bool(skip_existing),
            overwrite=bool(args.overwrite),
            model=str(args.model) if args.model else None,
            detect_config=str(args.detect_config) if args.detect_config else None,
            model_source=str(args.model_source),
            import_stimulus=bool(args.import_stimulus),
            refine_detect=bool(args.refine_detect),
            register=bool(args.register),
            registry=str(registry_path),
            set_id=str(args.set_id) if args.set_id else None,
            require_unique=bool(args.require_unique),
            include_non_success=bool(args.include_non_success),
            top_k=int(args.top_k) if args.top_k is not None else None,
        )

    plans = _build_plans(
        root,
        recursive=bool(args.recursive),
        skip_existing=skip_existing,
        check_stimulus=bool(args.import_stimulus),
    )

    if args.dry_run:
        print("Planned analysis creation (dry-run):")
        _print_plan(plans)
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
                    stimulus_present=plan.stimulus_present,
                )
            logger.log("run_end", ok=0, failed=0, skipped=0, missing=0, dry_run=True)
            logger.close()
        return 0

    registry: Optional[Registry] = None
    if args.register:
        registry = Registry(registry_path)
        print(f"Registry: {registry_path}")

    pipeline_opts = RecordingPipelineOptions(
        model_source=str(args.model_source),
        model=args.model,
        detect_config=args.detect_config,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch_size=args.batch_size,
        cpu=bool(args.cpu),
        set_id=args.set_id,
        require_unique=bool(args.require_unique),
        include_non_success=bool(args.include_non_success),
        top_k=int(args.top_k),
        refine_detect=bool(args.refine_detect),
        refine_config=args.refine_config,
        refine_max_gap=args.refine_max_gap,
        register=bool(args.register),
        registry_path=registry_path,
        import_opts=RecordingImportOptions(
            import_video_metadata=True,
            video_metadata_overwrite=False,
            import_stimulus=bool(args.import_stimulus),
            stimulus_always=bool(args.stimulus_always),
            stimulus_run_name=args.stimulus_run_name,
            stimulus_overwrite=bool(args.stimulus_overwrite),
            stimulus_quiet=bool(args.stimulus_quiet),
        ),
    )

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

        if plan.status == "skipped" and skip_existing:
            skipped += 1
            print(f"Skipping (exists): {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "recording_skipped",
                    recording_dir=str(plan.recording_dir),
                    zarr_path=str(plan.zarr_path),
                    reason=plan.reason,
                    status="skipped",
                )
            continue

        if logger is not None:
            logger.log(
                "recording_start",
                recording_dir=str(plan.recording_dir),
                h5_path=str(plan.h5_path),
                cam_video=str(plan.cam_video),
                zarr_path=str(plan.zarr_path),
            )

        assert plan.cam_video is not None
        result = process_recording_analysis_pipeline(
            SingleRecordingPlan(
                recording_dir=plan.recording_dir,
                h5_path=plan.h5_path,
                cam_video=plan.cam_video,
                zarr_path=plan.zarr_path,
            ),
            pipeline_opts,
            registry=registry,
            logger=(logger.log if logger is not None else None),
        )
        if not result.ok:
            failed += 1
            step = result.failed_step or "unknown_step"
            message = result.error or "step failed"
            print(f"{step} failed for {plan.zarr_path}: {message}")
            if logger is not None:
                logger.log(
                    "recording_failed",
                    recording_dir=str(plan.recording_dir),
                    zarr_path=str(plan.zarr_path),
                    step=step,
                    error=result.error,
                    returncode=result.returncode,
                )
            continue

        ok += 1
        if logger is not None:
            logger.log(
                "recording_ok",
                recording_dir=str(plan.recording_dir),
                zarr_path=str(plan.zarr_path),
                dataset_id=result.dataset_id,
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
            dry_run=False,
        )
        logger.close()

    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
