#!/usr/bin/env python3
"""Batch-run registry-backed detection on analysis zarr archives."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from fisheye.cli.shared_args import add_apply_dry_run_args
from fisheye.cli.shared_args import add_log_args
from fisheye.cli.shared_args import add_registry_discovery_args
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.environment import resolve_log_dir
from fisheye.shared.environment import resolve_recording_roots
from fisheye.shared.zarr_discovery import discover_registry_zarrs
from fisheye.utils.batch_registry_model_resolution import ResolvedModel
from fisheye.utils.batch_registry_model_resolution import (
    resolve_registry_models_for_plans as resolve_shared_registry_models_for_plans,
)
from fisheye.utils.resolve_detect_model import _load_candidates, _load_target_profile, _resolve_recording_id
from fisheye.utils.run_detect_with_registry_model import DetectRegistryResult
from fisheye.utils.run_detect_with_registry_model import _build_payload_args as _build_detect_payload_args
from fisheye.utils.run_detect_with_registry_model import _pick_best_candidate as _pick_detect_candidate
from fisheye.utils.run_detect_with_registry_model import (
    _resolution_payload as _build_detect_resolution_payload,
)
from fisheye.utils.run_detect_with_registry_model import (
    _write_model_resolution_provenance as _write_detect_model_resolution_provenance,
)
from fisheye.utils.zarr_recording_context import infer_recording_context

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore


STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_MISSING = "missing"
STATUS_FAILED = "failed"

DECODE_BACKEND_CHOICES = (
    "auto",
    "pynvvc_nv12_rgb",
    "pynvvc_luma_rgb",
    "decord_gpu",
    "decord_cpu",
    "opencv",
)

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
    camera_id: Optional[str] = None
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


_run_id = make_run_id


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
    context = infer_recording_context(zarr_path)
    local_recording_dir = _infer_recording_dir(zarr_path)
    recording_dir = context.recording_dir
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

    video_path, video_reason = _resolve_video_path(local_recording_dir)
    if video_path is not None:
        recording_dir = local_recording_dir
    elif context.recording_dir != local_recording_dir:
        video_path, video_reason = _resolve_video_path(context.recording_dir)
        if video_path is not None:
            recording_dir = context.recording_dir
    if video_path is None and context.source_video_path is not None and context.source_video_path.exists():
        video_path = context.source_video_path
        recording_dir = context.recording_dir
        video_reason = None
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


def _apply_registry_prereq(
    plans: Sequence[DetectPlan],
    *,
    registry_path: Path,
    require_registry: bool = True,
) -> List[DetectPlan]:
    if not require_registry:
        return list(plans)
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
                camera_id=plan.camera_id,
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


def _resolve_registry_model_for_plan(
    *,
    plan: DetectPlan,
    registry_path: Path,
    set_id_filter: Optional[str],
    require_unique: bool,
    top_k: int,
    include_non_success: bool,
    config: Optional[str],
    conf: Optional[float],
    iou: Optional[float],
    max_det: Optional[int],
    batch_size: Optional[int],
    resize_dims: Optional[list[int]],
    imgsz: Optional[list[int]],
    decode_backend: Optional[str],
    cpu: bool,
) -> ResolvedModel:
    if plan.video_path is None:
        raise ValueError("detect registry resolution requires a camera video path")

    context = infer_recording_context(plan.zarr_path)

    registry = Registry(registry_path)
    try:
        recording_id = _resolve_recording_id(
            registry,
            recording_id=context.recording_id,
            recording_dir=context.recording_dir,
        )
        target = _load_target_profile(registry, recording_id)
        candidates = _load_candidates(
            registry,
            target=target,
            task="detect",
            set_id_filter=set_id_filter,
            include_non_success=include_non_success,
        )
    finally:
        registry.close()

    best = _pick_detect_candidate(candidates, require_unique=require_unique)
    payload_args = _build_detect_payload_args(
        set_id=set_id_filter,
        require_unique=bool(require_unique),
        top_k=int(top_k),
        include_non_success=bool(include_non_success),
        dry_run=False,
        cpu=bool(cpu),
        config=config,
        conf=conf,
        iou=iou,
        max_det=max_det,
        batch_size=batch_size,
        resize_dims=resize_dims,
        imgsz=imgsz,
        decode_backend=decode_backend,
    )
    payload = _build_detect_resolution_payload(
        args=payload_args,
        argv=None,
        recording_dir=plan.recording_dir,
        video_path=plan.video_path,
        output_path=plan.zarr_path,
        registry_path=registry_path,
        recording_id=recording_id,
        target=target,
        selected=best,
        candidates=candidates,
        top_k=int(top_k),
    )
    return ResolvedModel(model_path=str(best.model_path), payload=payload)


def _resolve_registry_models_for_plans(
    *,
    plans: Sequence[DetectPlan],
    registry_path: Path,
    set_id_filter: Optional[str],
    require_unique: bool,
    top_k: int,
    include_non_success: bool,
    config: Optional[str],
    conf: Optional[float],
    iou: Optional[float],
    max_det: Optional[int],
    batch_size: Optional[int],
    resize_dims: Optional[list[int]],
    imgsz: Optional[list[int]],
    decode_backend: Optional[str],
    cpu: bool,
    on_event: Optional[Callable[[dict[str, object]], None]] = None,
) -> tuple[dict[str, ResolvedModel], dict[str, str]]:
    def _resolve(plan: DetectPlan) -> ResolvedModel:
        return _resolve_registry_model_for_plan(
            plan=plan,
            registry_path=registry_path,
            set_id_filter=set_id_filter,
            require_unique=require_unique,
            top_k=top_k,
            include_non_success=include_non_success,
            config=config,
            conf=conf,
            iou=iou,
            max_det=max_det,
            batch_size=batch_size,
            resize_dims=resize_dims,
            imgsz=imgsz,
            decode_backend=decode_backend,
            cpu=cpu,
        )

    return resolve_shared_registry_models_for_plans(
        plans=plans,
        resolve_plan=_resolve,
        on_event=on_event,
    )


def _build_explicit_model_payload(
    *,
    plan: DetectPlan,
    model_path: Path,
    config: Optional[str],
    conf: Optional[float],
    iou: Optional[float],
    max_det: Optional[int],
    batch_size: Optional[int],
    resize_dims: Optional[list[int]],
    imgsz: Optional[list[int]],
    decode_backend: Optional[str],
    cpu: bool,
) -> dict[str, object]:
    selected = {
        "model_path": str(model_path),
        "source": "explicit_cli",
        "run_id": None,
        "set_id": None,
    }
    return {
        "mode": "explicit",
        "task": "detect",
        "selected": selected,
        "candidates": [selected],
        "parameters": {
            "config": config,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "batch_size": batch_size,
            "resize_dims": resize_dims,
            "imgsz": imgsz,
            "decode_backend": decode_backend,
            "cpu": bool(cpu),
        },
        "inputs": {
            "recording_dir": str(plan.recording_dir),
            "video_path": str(plan.video_path) if plan.video_path else None,
            "output_zarr": str(plan.zarr_path),
        },
        "artifacts": {
            "selected_model": selected,
        },
    }


def _resolve_explicit_models_for_plans(
    *,
    plans: Sequence[DetectPlan],
    model_path: Path,
    config: Optional[str],
    conf: Optional[float],
    iou: Optional[float],
    max_det: Optional[int],
    batch_size: Optional[int],
    resize_dims: Optional[list[int]],
    imgsz: Optional[list[int]],
    decode_backend: Optional[str],
    cpu: bool,
) -> dict[str, ResolvedModel]:
    return {
        str(plan.zarr_path.resolve()): ResolvedModel(
            model_path=str(model_path),
            payload=_build_explicit_model_payload(
                plan=plan,
                model_path=model_path,
                config=config,
                conf=conf,
                iou=iou,
                max_det=max_det,
                batch_size=batch_size,
                resize_dims=resize_dims,
                imgsz=imgsz,
                decode_backend=decode_backend,
                cpu=cpu,
            ),
        )
        for plan in plans
    }


def _run_detect_plan(
    *,
    plan: DetectPlan,
    resolved_model: ResolvedModel,
    write_raw_video_metadata: bool,
    overwrite_raw_video_metadata: bool,
    config: Optional[str],
    conf: Optional[float],
    iou: Optional[float],
    max_det: Optional[int],
    batch_size: Optional[int],
    resize_dims: Optional[list[int]],
    imgsz: Optional[list[int]],
    decode_backend: Optional[str],
    cpu: bool,
    registry_path: Path,
) -> DetectRegistryResult:
    if plan.video_path is None:
        return DetectRegistryResult(
            ok=False,
            status="failed",
            reason="video_resolution_failed",
            error="camera video path missing",
            remediation="Ensure exactly one cams/*.mp4 exists or pass an explicit video path.",
            recording_dir=str(plan.recording_dir),
            output_zarr=str(plan.zarr_path),
            registry_path=str(registry_path),
            selected_model_path=resolved_model.model_path,
            resolved_at_utc=resolved_model.payload.get("resolved_at_utc"),
            resolution_payload=resolved_model.payload,
        )

    selected = resolved_model.payload.get("selected", {})
    selected_payload = selected if isinstance(selected, dict) else {}
    try:
        from fisheye.detection.detect_yolo import detect_yolo

        run_name = detect_yolo(
            video_path=str(plan.video_path),
            model_path=resolved_model.model_path,
            output_zarr=str(plan.zarr_path),
            config_path=config,
            conf_threshold=conf,
            iou_threshold=iou,
            max_det=max_det,
            batch_size=batch_size,
            resize_dims=resize_dims,
            imgsz=imgsz,
            decode_backend=decode_backend,
            use_gpu=(False if cpu else None),
            write_raw_video_metadata=bool(write_raw_video_metadata),
            overwrite_raw_video_metadata=bool(overwrite_raw_video_metadata),
        )
        if resolved_model.payload.get("mode") == "registry":
            _write_detect_model_resolution_provenance(
                zarr_path=plan.zarr_path,
                run_name=run_name,
                payload=resolved_model.payload,
            )
    except Exception as exc:
        return DetectRegistryResult(
            ok=False,
            status="failed",
            reason="detect_inference_failed",
            error=str(exc),
            remediation="Inspect model/config inputs and rerun with --dry-run --json to verify resolved model selection.",
            recording_dir=str(plan.recording_dir),
            output_zarr=str(plan.zarr_path),
            registry_path=str(registry_path),
            video_path=str(plan.video_path),
            selected_model_path=resolved_model.model_path,
            selected_run_id=selected_payload.get("run_id"),
            selected_set_id=selected_payload.get("set_id"),
            resolved_at_utc=resolved_model.payload.get("resolved_at_utc"),
            resolution_payload=resolved_model.payload,
        )

    return DetectRegistryResult(
        ok=True,
        status="ok",
        recording_dir=str(plan.recording_dir),
        output_zarr=str(plan.zarr_path),
        registry_path=str(registry_path),
        video_path=str(plan.video_path),
        selected_model_path=resolved_model.model_path,
        selected_run_id=selected_payload.get("run_id"),
        selected_set_id=selected_payload.get("set_id"),
        detect_run=run_name,
        resolved_at_utc=resolved_model.payload.get("resolved_at_utc"),
        resolution_payload=resolved_model.payload,
    )


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
    add_registry_discovery_args(
        parser,
        source_help="Discovery source for analysis zarrs (default: filesystem).",
        registry_help="Optional registry sqlite path.",
        rig_help="Filter by rig_id (registry source only).",
        arena_help="Filter by arena_id (registry source only).",
        camera_help="Filter by camera_id (registry source only).",
        path_contains_help="Substring match on zarr_path (registry source only).",
    )
    add_apply_dry_run_args(
        parser,
        apply_help="Run detect for planned analysis zarrs.",
        dry_run_help="Show planned detections without running.",
    )

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

    parser.add_argument("--set-id", type=str, help="Optional detect set filter during model resolution.")
    parser.add_argument("--require-unique", action="store_true", help="Fail if top model scores tie.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate models to persist in provenance.")
    parser.add_argument("--include-non-success", action="store_true", help="Allow non-success model rows in selection.")
    parser.add_argument(
        "--model",
        type=Path,
        help="Explicit YOLO detect model path. Bypasses registry model resolution.",
    )
    parser.add_argument(
        "--resolve-models",
        action="store_true",
        help="Resolve selected model paths during dry-run planning JSON output.",
    )

    parser.add_argument("--config", type=str, default=None, help="Optional detect config path.")
    parser.add_argument("--conf", type=float, default=None, help="Optional detect confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="Optional detect IoU threshold override.")
    parser.add_argument("--max-det", type=int, default=None, help="Optional detect max_det override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional detect batch size override.")
    parser.add_argument(
        "--resize-dims",
        nargs="+",
        type=int,
        default=None,
        help="Canonical inference size override [h w] (or one value for square); mapped to YOLO imgsz.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="Legacy alias for YOLO inference size; normalized into --resize-dims.",
    )
    parser.add_argument(
        "--decode-backend",
        choices=DECODE_BACKEND_CHOICES,
        default=None,
        help="Video decode backend passed to detect_yolo.",
    )
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
    add_log_args(
        parser,
        log_dir_help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/run_detections_batch or <root>/logs/run_detections_batch).",
    )

    args = parser.parse_args(argv)

    inputs = _resolve_input_paths(args.paths, args.file_list or [])
    registry_path = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    explicit_model_path: Optional[Path] = None
    if args.model is not None:
        explicit_model_path = args.model.expanduser()
        if args.apply and not explicit_model_path.exists():
            print(f"Explicit model does not exist: {explicit_model_path}", file=sys.stderr)
            return 1
        explicit_model_path = explicit_model_path.resolve()

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
                model_source=("explicit" if explicit_model_path is not None else "registry"),
                explicit_model=str(explicit_model_path) if explicit_model_path is not None else None,
                set_id=args.set_id,
                require_unique=bool(args.require_unique),
                include_non_success=bool(args.include_non_success),
                top_k=int(args.top_k),
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                batch_size=args.batch_size,
                resize_dims=args.resize_dims,
                imgsz=args.imgsz,
                decode_backend=args.decode_backend,
                cpu=bool(args.cpu),
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
    plans = _apply_registry_prereq(
        plans,
        registry_path=registry_path,
        require_registry=(explicit_model_path is None),
    )

    if not args.apply:
        console = Console() if Console is not None else None
        if console is not None:
            console.rule("[bold yellow]Dry run[/bold yellow]")
            console.print("Add [cyan]--apply[/cyan] to run detection.")
        else:
            print("Dry run: add --apply to run detection.")

        dry_run_resolved_models: dict[str, ResolvedModel] = {}
        dry_run_resolution_errors: dict[str, str] = {}
        if args.resolve_models:
            runnable_dry_run_plans = [plan for plan in plans if plan.status == STATUS_OK]
            if explicit_model_path is not None:
                dry_run_resolved_models = _resolve_explicit_models_for_plans(
                    plans=runnable_dry_run_plans,
                    model_path=explicit_model_path,
                    config=args.config,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    batch_size=args.batch_size,
                    resize_dims=args.resize_dims,
                    imgsz=args.imgsz,
                    decode_backend=args.decode_backend,
                    cpu=bool(args.cpu),
                )
            elif runnable_dry_run_plans:
                dry_run_resolved_models, dry_run_resolution_errors = _resolve_registry_models_for_plans(
                    plans=runnable_dry_run_plans,
                    registry_path=registry_path,
                    set_id_filter=args.set_id,
                    require_unique=bool(args.require_unique),
                    top_k=int(args.top_k),
                    include_non_success=bool(args.include_non_success),
                    config=args.config,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    batch_size=args.batch_size,
                    resize_dims=args.resize_dims,
                    imgsz=args.imgsz,
                    decode_backend=args.decode_backend,
                    cpu=bool(args.cpu),
                    on_event=None,
                )

        if args.json:
            for plan in plans:
                payload = _plan_payload(plan)
                if args.resolve_models and plan.status == STATUS_OK:
                    plan_key = str(plan.zarr_path.resolve())
                    resolved_model = dry_run_resolved_models.get(plan_key)
                    if resolved_model is not None:
                        selected = resolved_model.payload.get("selected")
                        selected_payload = selected if isinstance(selected, dict) else {}
                        payload.update(
                            {
                                "model_resolution_status": STATUS_OK,
                                "selected_model": resolved_model.model_path,
                                "selected_run_id": selected_payload.get("run_id"),
                                "selected_set_id": selected_payload.get("set_id"),
                            }
                        )
                    else:
                        payload.update(
                            {
                                "model_resolution_status": STATUS_FAILED,
                                "model_resolution_error": dry_run_resolution_errors.get(
                                    plan_key,
                                    "model not resolved",
                                ),
                            }
                        )
                print(json.dumps(payload, sort_keys=True))
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

    runnable_plans: list[DetectPlan] = []
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
        runnable_plans.append(plan)

    resolved_models_by_plan: dict[str, ResolvedModel] = {}
    if runnable_plans:
        if explicit_model_path is not None:
            if not args.json:
                print(
                    "Using explicit detect model for "
                    f"{len(runnable_plans)} recording plan(s): {explicit_model_path}"
                )
            resolved_models_by_plan = _resolve_explicit_models_for_plans(
                plans=runnable_plans,
                model_path=explicit_model_path,
                config=args.config,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                batch_size=args.batch_size,
                resize_dims=args.resize_dims,
                imgsz=args.imgsz,
                decode_backend=args.decode_backend,
                cpu=bool(args.cpu),
            )
            if logger is not None:
                logger.log(
                    "model_resolution_summary",
                    zarr="-",
                    status=STATUS_OK,
                    mode="explicit",
                    selected_model_path=str(explicit_model_path),
                    total=len(runnable_plans),
                    resolved=len(resolved_models_by_plan),
                    failed=0,
                )
        elif not args.json:
            print(f"Pre-resolving registry models for {len(runnable_plans)} recording plan(s)...")

    if runnable_plans and explicit_model_path is None:
        def _emit_model_resolution_event(event_payload: dict[str, object]) -> None:
            event_name = str(event_payload.get("event", "model_resolution_event"))
            if logger is not None:
                logger.log(
                    event_name,
                    zarr=str(event_payload.get("zarr")),
                    status=STATUS_OK,
                    **{k: v for k, v in event_payload.items() if k not in {"event", "zarr"}},
                )
            if args.json:
                return
            index = event_payload.get("index")
            total = event_payload.get("total")
            prefix = f"[{index}/{total}] " if index is not None and total is not None else ""
            if event_name == "model_resolution_start":
                print(f"{prefix}Resolving model for {event_payload.get('recording')}")
                return
            if event_name == "model_resolution_cached":
                print(
                    f"{prefix}Using cached model for {event_payload.get('recording')}: "
                    f"{event_payload.get('selected_model_path')}"
                )
                return
            if event_name == "model_resolution_ok":
                print(
                    f"{prefix}Resolved model for {event_payload.get('recording')}: "
                    f"{event_payload.get('selected_model_path')}"
                )
                return
            if event_name == "model_resolution_failed":
                print(
                    f"{prefix}Model resolution failed for {event_payload.get('zarr')}: "
                    f"{event_payload.get('error')}"
                )

        resolved_models_by_plan, resolution_errors = _resolve_registry_models_for_plans(
            plans=runnable_plans,
            registry_path=registry_path,
            set_id_filter=args.set_id,
            require_unique=bool(args.require_unique),
            top_k=int(args.top_k),
            include_non_success=bool(args.include_non_success),
            config=args.config,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            resize_dims=args.resize_dims,
            imgsz=args.imgsz,
            decode_backend=args.decode_backend,
            cpu=bool(args.cpu),
            on_event=_emit_model_resolution_event,
        )
        if logger is not None:
            logger.log(
                "model_resolution_summary",
                zarr="-",
                status=STATUS_OK,
                total=len(runnable_plans),
                resolved=len(resolved_models_by_plan),
                failed=len(resolution_errors),
            )
        if not args.json:
            print(
                "Model resolution summary: "
                f"resolved={len(resolved_models_by_plan)} failed={len(resolution_errors)}"
            )
        if resolution_errors:
            runnable_after_resolution: list[DetectPlan] = []
            for plan in runnable_plans:
                key = str(plan.zarr_path.resolve())
                error = resolution_errors.get(key)
                if error is None:
                    runnable_after_resolution.append(plan)
                    continue
                failed += 1
                if args.json:
                    print(
                        json.dumps(
                            {
                                "recording": str(plan.recording_dir),
                                "zarr": str(plan.zarr_path),
                                "status": STATUS_FAILED,
                                "reason": "model_resolution_failed",
                                "error": error,
                            },
                            sort_keys=True,
                        )
                    )
                else:
                    print(f"Detection failed for {plan.zarr_path}: model resolution failed ({error})")
                if logger is not None:
                    logger.log(
                        "detect_failed",
                        zarr=str(plan.zarr_path),
                        status=STATUS_FAILED,
                        reason="model_resolution_failed",
                        error=error,
                    )
            runnable_plans = runnable_after_resolution

    for plan in runnable_plans:
        result = _run_detect_plan(
            plan=plan,
            resolved_model=resolved_models_by_plan[str(plan.zarr_path.resolve())],
            write_raw_video_metadata=bool(args.write_raw_video_metadata),
            overwrite_raw_video_metadata=bool(args.overwrite_raw_video_metadata),
            config=args.config,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            resize_dims=args.resize_dims,
            imgsz=args.imgsz,
            decode_backend=args.decode_backend,
            cpu=bool(args.cpu),
            registry_path=registry_path,
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
