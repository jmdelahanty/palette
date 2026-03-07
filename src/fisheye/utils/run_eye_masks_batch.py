import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import h5py
import yaml
import zarr

from fisheye.cli.shared_args import add_apply_dry_run_args
from fisheye.cli.shared_args import add_log_args
from fisheye.cli.shared_args import add_registry_discovery_args
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.status_ledger import upsert_recording_step_status
from fisheye.refinement.refine_eye_masks import refine_eye_masks
from fisheye.segmentation.eye_segmentation import segment_eye_masks as segment_eye_masks_traditional
from fisheye.segmentation.eye_segmentation_yolo import segment_eye_masks_yolo
from fisheye.segmentation.infer_unet_eye_masks import main as infer_unet_eye_masks_main
from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.environment import resolve_log_dir as resolve_shared_log_dir
from fisheye.shared.environment import resolve_recording_roots as resolve_shared_recording_roots
from fisheye.shared.provenance_attrs import resolve_source_keypoints_run
from fisheye.shared.registry_stage_complete import DatasetMetadata, emit_stage_completion
from fisheye.shared.type_conversions import normalize_attr as _normalize_attr
from fisheye.shared.zarr_discovery import discover_registry_zarrs as discover_shared_registry_zarrs
from fisheye.utils.batch_registry_model_resolution import ResolvedModel
from fisheye.utils.batch_registry_model_resolution import (
    resolve_registry_models_for_plans as resolve_shared_registry_models_for_plans,
)
from fisheye.utils.resolve_detect_model import _load_candidates, _load_target_profile, _resolve_recording_id
from fisheye.utils.zarr_recording_context import infer_recording_context
from fisheye.utils.run_eye_masks_with_registry_model import _pick_best_candidate as _pick_eye_mask_candidate
from fisheye.utils.run_eye_masks_with_registry_model import (
    _resolution_payload as _build_eye_mask_resolution_payload,
)
from fisheye.utils.run_eye_masks_with_registry_model import (
    _write_model_resolution_provenance as _write_eye_mask_model_resolution_provenance,
)
from fisheye.utils.run_eye_masks_with_registry_model import main as run_eye_masks_with_registry_model_main

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
class EyeMaskPlan:
    recording_dir: Path
    h5_path: Optional[Path]
    zarr_path: Path
    camera_id: Optional[str]
    status: str
    reason: Optional[str] = None
    eye_masks_present: bool = False
    refined_eye_masks_present: bool = False
    crop_present: bool = False
    keypoints_present: bool = False


_utc_now = utc_now


JsonLogger = SharedJsonLogger


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
    return resolve_shared_recording_roots(paths)


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    return resolve_shared_log_dir(arg_log_dir, roots, log_subdir="run_eye_masks_batch")


_run_id = make_run_id


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
    lines = path.read_text(encoding="utf-8").splitlines()
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


def _infer_zarr_use_from_name(zarr_path: Path) -> Optional[str]:
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        value = _normalize_attr(root.attrs.get(key))
        if value in {"analysis", "training"}:
            return value
    return _infer_zarr_use_from_name(zarr_path)


def _zarr_matches_target_use(zarr_path: Path, target_use: str) -> bool:
    if target_use == "all":
        return True
    hinted = _infer_zarr_use_from_name(zarr_path)
    if hinted is not None:
        return hinted == target_use
    if not zarr_path.exists():
        return True
    try:
        root = zarr.open(str(zarr_path), mode="r", use_consolidated=False)
    except Exception:
        return True
    inferred = _infer_zarr_use(root, zarr_path)
    if inferred is None:
        return True
    return inferred == target_use


def _normalize_h5_stem(stem: str) -> str:
    lowered = stem.lower()
    if lowered.endswith("_analysis"):
        return stem[: -len("_analysis")]
    if lowered.endswith("_training"):
        return stem[: -len("_training")]
    return stem


def _candidate_zarr_paths_for_h5(recording_dir: Path, h5_stem: str, target_use: str) -> List[Path]:
    base_stem = _normalize_h5_stem(h5_stem)
    names: List[str] = []
    if target_use in {"all", "analysis"}:
        names.append(f"{base_stem}_analysis.zarr")
    if target_use in {"all", "training"}:
        names.append(f"{base_stem}_training.zarr")
    names.extend([f"{h5_stem}.zarr", f"{base_stem}.zarr"])

    unique_names: List[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)
    return [recording_dir / "zarr" / name for name in unique_names]


def _has_latest_run(root: zarr.Group, group_name: str) -> bool:
    parent = root.get(group_name)
    if parent is None:
        return False
    return bool(parent.attrs.get("latest"))


def _has_crop(root: zarr.Group) -> bool:
    return _has_latest_run(root, "crop_runs")


def _has_any_keypoints(root: zarr.Group) -> bool:
    return _has_latest_run(root, "keypoints_runs") or _has_latest_run(root, "refined_keypoints_runs")


def _plan_from_zarr(
    *,
    zarr_path: Path,
    recording_dir: Path,
    h5_path: Optional[Path],
    camera_id: Optional[str],
    skip_existing: bool,
    require_crop: bool,
    require_keypoints: bool,
    refine_only: bool,
    refine_missing_only: bool = False,
) -> EyeMaskPlan:
    if not zarr_path.exists():
        return EyeMaskPlan(
            recording_dir=recording_dir,
            h5_path=h5_path,
            zarr_path=zarr_path,
            camera_id=camera_id,
            status="missing",
            reason="zarr missing",
        )
    try:
        root = zarr.open(str(zarr_path), mode="r", use_consolidated=False)
    except Exception as exc:
        return EyeMaskPlan(
            recording_dir=recording_dir,
            h5_path=h5_path,
            zarr_path=zarr_path,
            camera_id=camera_id,
            status="missing",
            reason=f"zarr open failed: {exc}",
        )

    crop_present = _has_crop(root)
    keypoints_present = _has_any_keypoints(root)
    eye_masks_present = _has_latest_run(root, "eye_masks_runs")
    refined_eye_masks_present = _has_latest_run(root, "refined_eye_masks_runs")

    if refine_only:
        if not eye_masks_present:
            return EyeMaskPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="missing",
                reason="eye_masks missing",
                eye_masks_present=eye_masks_present,
                refined_eye_masks_present=refined_eye_masks_present,
                crop_present=crop_present,
                keypoints_present=keypoints_present,
            )
        if refine_missing_only and refined_eye_masks_present:
            return EyeMaskPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="skipped",
                reason="refined eye masks already present",
                eye_masks_present=eye_masks_present,
                refined_eye_masks_present=refined_eye_masks_present,
                crop_present=crop_present,
                keypoints_present=keypoints_present,
            )
    else:
        if require_crop and not crop_present:
            return EyeMaskPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="missing",
                reason="crop missing",
                eye_masks_present=eye_masks_present,
                refined_eye_masks_present=refined_eye_masks_present,
                crop_present=crop_present,
                keypoints_present=keypoints_present,
            )
        if require_keypoints and not keypoints_present:
            return EyeMaskPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="missing",
                reason="keypoints missing",
                eye_masks_present=eye_masks_present,
                refined_eye_masks_present=refined_eye_masks_present,
                crop_present=crop_present,
                keypoints_present=keypoints_present,
            )
        if skip_existing and eye_masks_present:
            return EyeMaskPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=camera_id,
                status="skipped",
                reason="eye masks already present",
                eye_masks_present=eye_masks_present,
                refined_eye_masks_present=refined_eye_masks_present,
                crop_present=crop_present,
                keypoints_present=keypoints_present,
            )

    return EyeMaskPlan(
        recording_dir=recording_dir,
        h5_path=h5_path,
        zarr_path=zarr_path,
        camera_id=camera_id,
        status="ok",
        eye_masks_present=eye_masks_present,
        refined_eye_masks_present=refined_eye_masks_present,
        crop_present=crop_present,
        keypoints_present=keypoints_present,
    )


def _build_plans(
    roots: List[Path],
    recursive: bool,
    skip_existing: bool,
    require_crop: bool,
    require_keypoints: bool,
    refine_only: bool,
    zarr_use: str,
    refine_missing_only: bool = False,
) -> List[EyeMaskPlan]:
    plans: List[EyeMaskPlan] = []
    for h5_path in _iter_h5(roots, recursive):
        recording_dir = h5_path.parent.parent
        camera_id = _read_camera_id(h5_path)

        candidate_paths = _candidate_zarr_paths_for_h5(recording_dir, h5_path.stem, zarr_use)
        primary_count = 2 if zarr_use == "all" else 1
        existing_primary = [
            path
            for path in candidate_paths[:primary_count]
            if path.exists() and _zarr_matches_target_use(path, zarr_use)
        ]
        existing_fallback = [
            path
            for path in candidate_paths[primary_count:]
            if path.exists() and _zarr_matches_target_use(path, zarr_use)
        ]
        if existing_primary:
            selected_paths = existing_primary
        elif existing_fallback:
            selected_paths = existing_fallback
        else:
            selected_paths = candidate_paths[:1]

        for zarr_path in selected_paths:
            plans.append(
                _plan_from_zarr(
                    zarr_path=zarr_path,
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    camera_id=camera_id,
                    skip_existing=skip_existing,
                    require_crop=require_crop,
                    require_keypoints=require_keypoints,
                    refine_only=refine_only,
                    refine_missing_only=refine_missing_only,
                )
            )
    return plans


def _build_plans_from_zarr(
    zarr_paths: Iterable[Path],
    skip_existing: bool,
    require_crop: bool,
    require_keypoints: bool,
    refine_only: bool,
    zarr_use: str,
    refine_missing_only: bool = False,
) -> List[EyeMaskPlan]:
    plans: List[EyeMaskPlan] = []
    for zarr_path in zarr_paths:
        if not _zarr_matches_target_use(zarr_path, zarr_use):
            continue
        recording_dir = infer_recording_context(zarr_path).recording_dir
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
                require_keypoints=require_keypoints,
                refine_only=refine_only,
                refine_missing_only=refine_missing_only,
            )
        )
    return plans


def _discover_zarrs_from_registry(
    *,
    registry_path: Path,
    scope_paths: Sequence[Path],
    rig_id: Optional[str] = None,
    arena_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    path_contains: Optional[str] = None,
    skip_existing: bool = False,
) -> List[Path]:
    """Query the registry for analysis zarr paths suitable for eye mask processing.

    When *skip_existing* is True, recordings whose ``eye_masks`` step status is
    already ``'ok'`` are excluded at the SQL level.  Only recordings whose
    ``crop`` and ``keypoints`` steps are both ``'ok'`` are returned, ensuring
    prerequisite pipeline stages are complete.
    """
    return discover_shared_registry_zarrs(
        registry_path=registry_path,
        scope_paths=scope_paths,
        zarr_use="analysis",
        rig_id=rig_id,
        arena_id=arena_id,
        camera_id=camera_id,
        path_contains=path_contains,
        require_steps_ok=["crop", "keypoints"],
        exclude_step_ok="eye_masks" if skip_existing else None,
        zarr_suffix="_analysis.zarr",
    )


def _print_plan(plans: List[EyeMaskPlan]) -> None:
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
        return _canonical_method(str(override).lower())
    method = config.get("eye_masks", {}).get("method", "traditional")
    return _canonical_method(str(method).lower())


def _canonical_method(method: str) -> str:
    alias_map = {
        "traditional": "traditional",
        "yolo": "yolo",
        "yolo_eye_segmentation": "yolo",
        "unet": "unet",
        "unet_eye_masks": "unet",
    }
    canonical = alias_map.get(str(method).lower())
    if canonical is None:
        raise ValueError(
            f"Unsupported eye mask method '{method}'. Expected one of: traditional, yolo, unet."
        )
    return canonical


def _validate_method_requirements(config: Dict[str, Any], method: str, *, refine_only: bool) -> None:
    if refine_only:
        return
    params = config.get("eye_masks", {}) or {}
    canonical_method = _canonical_method(method)
    if canonical_method == "yolo":
        model_path = params.get("model_path") or params.get("model")
        if not model_path:
            raise ValueError("Method 'yolo' requires eye_masks.model_path or eye_masks.model in config.")
        return
    if canonical_method == "unet":
        checkpoint = (
            params.get("checkpoint")
            or params.get("checkpoint_path")
            or params.get("model_path")
            or params.get("model")
        )
        if not checkpoint:
            raise ValueError(
                "Method 'unet' requires eye_masks.checkpoint, checkpoint_path, model_path, or model in config."
            )
        return


def _latest_run(zarr_path: Path, group_name: str) -> Optional[str]:
    try:
        root = zarr.open(str(zarr_path), mode="r", use_consolidated=False)
    except Exception:
        return None
    parent = root.get(group_name)
    if parent is None:
        return None
    latest = parent.attrs.get("latest")
    return str(latest) if latest else None


def _decode_source_runs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    source_runs: Dict[str, Any] = {}
    for key in (
        "source_crop_run",
        "source_detect_run",
        "source_background_run",
        "source_eye_masks_run",
        "source_eye_masks_method",
        "source_keypoint_group",
    ):
        if key in attrs:
            source_runs[key] = attrs.get(key)
    source_kp = resolve_source_keypoints_run(attrs)
    if source_kp is not None:
        source_runs["source_keypoints_run"] = source_kp
    return source_runs


def _resolve_step_status_context(
    zarr_path: Path,
    *,
    registry_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    resolved_zarr = zarr_path.expanduser().resolve()
    resolved_registry_path = (
        registry_path.expanduser().resolve()
        if registry_path is not None
        else RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    )
    if not resolved_registry_path.exists():
        return None

    registry = Registry(resolved_registry_path)
    try:
        path_hash = hashlib.sha256(str(resolved_zarr).encode("utf-8")).hexdigest()
        row = registry.conn.execute(
            """
            SELECT dataset_id, recording_id
            FROM datasets
            WHERE path_hash = ?
            ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
            LIMIT 1;
            """,
            (path_hash,),
        ).fetchone()
        if row is None:
            row = registry.conn.execute(
                """
                SELECT dataset_id, recording_id
                FROM datasets
                WHERE zarr_path = ?
                ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
                LIMIT 1;
                """,
                (str(resolved_zarr),),
            ).fetchone()
        if row is None:
            row = registry.conn.execute(
                """
                SELECT dataset_id, recording_id
                FROM datasets
                WHERE lower(COALESCE(zarr_path, '')) = lower(?)
                ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
                LIMIT 1;
                """,
                (str(resolved_zarr),),
            ).fetchone()
        if row is None:
            return None
        dataset_id = _normalize_attr(row["dataset_id"])
        if not dataset_id:
            return None
        return {
            "registry_path": resolved_registry_path,
            "dataset_id": dataset_id,
            "recording_id": _normalize_attr(row["recording_id"]),
        }
    finally:
        registry.close()


def _zarr_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return None


def _coverage_pct_from_stage_payload(stage_payload: Dict[str, Any]) -> Optional[float]:
    for key in ("successful_roi_pair_rate", "success_rate", "coverage_pct"):
        if key not in stage_payload:
            continue
        try:
            value = float(stage_payload[key])
        except (TypeError, ValueError):
            continue
        if value <= 1.0:
            value *= 100.0
        return value

    summary = stage_payload.get("summary_statistics")
    if isinstance(summary, dict):
        for key in ("successful_roi_pair_rate", "success_rate_percent", "pass_rate_percent"):
            if key not in summary:
                continue
            try:
                value = float(summary[key])
            except (TypeError, ValueError):
                continue
            if value <= 1.0:
                value *= 100.0
            return value
    return None


def _step_status_details_from_stage_payload(
    stage_payload: Dict[str, Any],
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "reason": "present",
        "latest_selector": "run_eye_masks_batch",
    }
    source_runs = stage_payload.get("source_runs")
    if isinstance(source_runs, dict):
        for key in (
            "source_crop_run",
            "source_detect_run",
            "source_background_run",
            "source_eye_masks_run",
            "source_eye_masks_method",
            "source_keypoint_group",
            "source_keypoints_run",
        ):
            if source_runs.get(key) is not None:
                details[key] = source_runs.get(key)

    for key in ("total_rois", "successful_roi_pairs", "successful_roi_pair_rate"):
        if stage_payload.get(key) is not None:
            details[key] = stage_payload.get(key)

    summary = stage_payload.get("summary_statistics")
    if isinstance(summary, dict):
        for key in ("total_rois", "successful_roi_pairs", "successful_roi_pair_rate"):
            if summary.get(key) is not None and key not in details:
                details[key] = summary.get(key)
    return details


def _sync_eye_mask_registry_rows_after_run(
    *,
    zarr_path: Path,
    stage_payloads: Dict[str, Dict[str, Any]],
    registry_path: Optional[Path] = None,
) -> Dict[str, Any]:
    context = _resolve_step_status_context(zarr_path, registry_path=registry_path)
    if context is None:
        return {"synced": False, "reason": "status_context_unavailable"}

    registry = Registry(Path(str(context["registry_path"])))
    try:
        dataset_id = str(context["dataset_id"])
        dataset_row = registry.conn.execute(
            """
            SELECT recording_id, zarr_use
            FROM datasets
            WHERE dataset_id = ?
            LIMIT 1;
            """,
            (dataset_id,),
        ).fetchone()
        recording_id = (
            _normalize_attr(dataset_row["recording_id"])
            if dataset_row is not None
            else _normalize_attr(context.get("recording_id"))
        )
        zarr_use = _normalize_attr(dataset_row["zarr_use"]) if dataset_row is not None else None
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            session_uuid=None,
            recording_id=recording_id,
            zarr_use=zarr_use,
            zarr_purpose=None,
        )

        step_status_written: List[str] = []
        step_status_errors: Dict[str, str] = {}
        for step_name, payload in stage_payloads.items():
            if not isinstance(payload, dict):
                continue
            run_name = _normalize_attr(payload.get("run_name"))
            if not run_name:
                continue
            review_status = payload.get("review_status")
            try:
                def _upsert_with_mtime(*args: Any, **kwargs: Any) -> None:
                    kwargs["zarr_mtime_ns"] = _zarr_mtime_ns(zarr_path)
                    upsert_recording_step_status(*args, **kwargs)

                wrote = emit_stage_completion(
                    None,
                    zarr_path,
                    step_name=step_name,
                    status="ok",
                    source="run_eye_masks_batch_postrun_sync",
                    run_name=run_name,
                    method=(
                        _normalize_attr(payload.get("method"))
                        or ("refine_eye_masks" if step_name == "refined_eye_masks" else "eye_masks")
                    ),
                    coverage_pct=_coverage_pct_from_stage_payload(payload),
                    review_status_json=review_status if isinstance(review_status, dict) else None,
                    details_json=_step_status_details_from_stage_payload(payload),
                    console=None,
                    registry=registry,
                    auto_registry_from_env=False,
                    invalidate_on_ok=False,
                    metadata=metadata,
                    upsert_dataset_row=False,
                    upsert_step_status_fn=_upsert_with_mtime,
                )
                if not wrote:
                    raise RuntimeError("emit_stage_completion returned false")
                step_status_written.append(step_name)
            except Exception as exc:  # pragma: no cover - defensive
                step_status_errors[step_name] = str(exc)

        perf_rows: Optional[int] = None
        quality_rows: Optional[int] = None
        refresh_error: Optional[str] = None
        try:
            perf_rows = int(
                registry.refresh_eye_mask_performance_for_dataset(
                    dataset_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    zarr_use=zarr_use,
                )
            )
            quality_rows = int(
                registry.refresh_eye_mask_quality_for_dataset(
                    dataset_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    zarr_use=zarr_use,
                )
            )
        except Exception as exc:
            refresh_error = str(exc)

        result: Dict[str, Any] = {
            "synced": not step_status_errors and refresh_error is None,
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "step_status_written": step_status_written,
            "step_status_errors": step_status_errors,
            "eye_mask_performance_rows": perf_rows,
            "eye_mask_quality_rows": quality_rows,
        }
        if refresh_error is not None:
            result["reason"] = "refresh_failed"
            result["error"] = refresh_error
        elif step_status_errors:
            result["reason"] = "step_status_upsert_failed"
        return result
    except Exception as exc:
        return {"synced": False, "reason": "refresh_failed", "error": str(exc)}
    finally:
        registry.close()


def _collect_stage_payload(zarr_path: Path, group_name: str, run_name: str) -> Dict[str, Any]:
    root = zarr.open(str(zarr_path), mode="r", use_consolidated=False)
    parent = root.get(group_name)
    if parent is None or run_name not in parent:
        raise RuntimeError(f"run not found after stage completion: {group_name}/{run_name}")
    run_group = parent[run_name]
    attrs = dict(run_group.attrs)
    summary = attrs.get("summary_statistics")
    summary_statistics = summary if isinstance(summary, dict) else {}
    return {
        "group": group_name,
        "run_name": run_name,
        "method": _normalize_attr(attrs.get("method")),
        "duration_seconds": attrs.get("duration_seconds"),
        "total_rois": attrs.get("total_rois"),
        "successful_roi_pairs": attrs.get("successful_roi_pairs"),
        "successful_roi_pair_rate": attrs.get("successful_roi_pair_rate"),
        "review_status": attrs.get("eye_mask_review_status"),
        "summary_statistics": summary_statistics,
        "source_runs": _decode_source_runs(attrs),
    }


def _run_traditional(
    zarr_path: Path,
    config: Dict[str, Any],
    scheduler: Optional[str],
    num_workers: Optional[int],
    quiet: bool,
) -> str:
    params = config.get("eye_masks", {}) or {}
    config_dict = {
        key: value
        for key, value in params.items()
        if key
        not in {
            "method",
            "model_path",
            "model",
            "run_name",
            "crop_run",
            "batch_size",
            "device",
            "imgsz",
            "conf",
            "iou",
            "max_det",
            "mask_threshold",
            "adaptive_scale",
            "adaptive_cap",
            "use_retina_masks",
            "proto_upsample_factor",
            "legacy_masks",
            "roi_cache_policy",
            "roi_cache_dir",
            "verbose",
        }
    }
    if quiet and Console is not None:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            console = Console(file=devnull, force_terminal=False)
            return segment_eye_masks_traditional(
                zarr_path=str(zarr_path),
                config_dict=config_dict,
                console=console,
                scheduler=scheduler or "threads",
                num_workers=num_workers,
            )
    return segment_eye_masks_traditional(
        zarr_path=str(zarr_path),
        config_dict=config_dict,
        console=None,
        scheduler=scheduler or "threads",
        num_workers=num_workers,
    )


def _run_yolo(
    zarr_path: Path,
    config: Dict[str, Any],
    quiet: bool,
    *,
    model_path_override: Optional[str] = None,
) -> str:
    params = config.get("eye_masks", {}) or {}
    model_path = model_path_override or params.get("model_path") or params.get("model")
    if not model_path:
        raise ValueError("YOLO eye masks require 'model_path' or 'model' in config.")
    if quiet and Console is not None:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            console = Console(file=devnull, force_terminal=False)
            return segment_eye_masks_yolo(
                zarr_path=str(zarr_path),
                model_path=model_path,
                run_name=params.get("run_name"),
                crop_run=params.get("crop_run"),
                keypoints_run=params.get("keypoints_run"),
                batch_size=params.get("batch_size", 128),
                device=params.get("device"),
                imgsz=params.get("imgsz"),
                conf=params.get("conf", 0.05),
                iou=params.get("iou", 0.5),
                max_det=params.get("max_det", 2),
                mask_threshold=params.get("mask_threshold", 0.05),
                adaptive_scale=params.get("adaptive_scale", 0.6),
                adaptive_cap=params.get("adaptive_cap", 0.6),
                use_retina_masks=params.get("use_retina_masks", True),
                proto_upsample_factor=params.get("proto_upsample_factor", 2),
                legacy_masks=params.get("legacy_masks", False),
                roi_cache_policy=params.get("roi_cache_policy", "auto"),
                roi_cache_dir=params.get("roi_cache_dir"),
                verbose=params.get("verbose", False),
                console=console,
            )
    return segment_eye_masks_yolo(
        zarr_path=str(zarr_path),
        model_path=model_path,
        run_name=params.get("run_name"),
        crop_run=params.get("crop_run"),
        keypoints_run=params.get("keypoints_run"),
        batch_size=params.get("batch_size", 128),
        device=params.get("device"),
        imgsz=params.get("imgsz"),
        conf=params.get("conf", 0.05),
        iou=params.get("iou", 0.5),
        max_det=params.get("max_det", 2),
        mask_threshold=params.get("mask_threshold", 0.05),
        adaptive_scale=params.get("adaptive_scale", 0.6),
        adaptive_cap=params.get("adaptive_cap", 0.6),
        use_retina_masks=params.get("use_retina_masks", True),
        proto_upsample_factor=params.get("proto_upsample_factor", 2),
        legacy_masks=params.get("legacy_masks", False),
        roi_cache_policy=params.get("roi_cache_policy", "auto"),
        roi_cache_dir=params.get("roi_cache_dir"),
        verbose=params.get("verbose", False),
        console=None,
    )


def _run_unet(
    zarr_path: Path,
    config: Dict[str, Any],
    *,
    checkpoint_override: Optional[str] = None,
) -> str:
    params = config.get("eye_masks", {}) or {}
    checkpoint = (
        checkpoint_override
        or params.get("checkpoint")
        or params.get("checkpoint_path")
        or params.get("model_path")
        or params.get("model")
    )
    if not checkpoint:
        raise ValueError("U-Net eye masks require 'checkpoint' or 'checkpoint_path' in config.")

    argv: List[str] = [str(zarr_path), "--checkpoint", str(checkpoint)]
    if params.get("batch_size") is not None:
        argv.extend(["--batch-size", str(params.get("batch_size"))])
    if params.get("device"):
        argv.extend(["--device", str(params.get("device"))])
    if params.get("label_mode"):
        argv.extend(["--label-mode", str(params.get("label_mode"))])
    if params.get("run_name"):
        argv.extend(["--run-name", str(params.get("run_name"))])
    if params.get("source_run"):
        argv.extend(["--eye-mask-run", str(params.get("source_run"))])
    if params.get("crop_run"):
        argv.extend(["--crop-run", str(params.get("crop_run"))])
    if params.get("keypoints_run"):
        argv.extend(["--keypoints-run", str(params.get("keypoints_run"))])
    if params.get("write_binary_masks"):
        argv.append("--write-binary-masks")
    if params.get("roi_cache_policy"):
        argv.extend(["--roi-cache-policy", str(params.get("roi_cache_policy"))])
    if params.get("roi_cache_dir"):
        argv.extend(["--roi-cache-dir", str(params.get("roi_cache_dir"))])
    if params.get("use_crop"):
        argv.append("--use-crop")

    infer_unet_eye_masks_main(argv)
    latest = _latest_run(zarr_path, "eye_masks_runs")
    if not latest:
        raise RuntimeError("U-Net eye-mask run completed but no latest eye_masks run found.")
    return latest


def _run_refine(zarr_path: Path, config: Dict[str, Any], source_run: Optional[str], quiet: bool) -> str:
    params = config.get("refine_eye_masks", {}) or {}
    scheduler = str(params.get("scheduler", "processes")).lower()
    if scheduler not in {"threads", "processes"}:
        scheduler = "processes"
    if quiet and Console is not None:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            console = Console(file=devnull, force_terminal=False)
            return refine_eye_masks(
                zarr_path=str(zarr_path),
                source_run=source_run or params.get("source_run"),
                run_name=params.get("run_name"),
                keypoint_run=params.get("keypoint_run"),
                chunk_size=int(params.get("chunk_size", 512)),
                scheduler=scheduler,
                num_workers=params.get("num_workers"),
                console=console,
                command="run_eye_masks_batch --refine",
                created_at_utc=_utc_now(),
                area_filter_z=params.get("area_filter_z", 2.0),
                area_filter_mode=params.get("area_filter_mode", "either"),
                force_refine_traditional=bool(params.get("force_refine_traditional", False)),
                allow_latest_keypoint_fallback=bool(params.get("allow_latest_keypoint_fallback", False)),
            )
    return refine_eye_masks(
        zarr_path=str(zarr_path),
        source_run=source_run or params.get("source_run"),
        run_name=params.get("run_name"),
        keypoint_run=params.get("keypoint_run"),
        chunk_size=int(params.get("chunk_size", 512)),
        scheduler=scheduler,
        num_workers=params.get("num_workers"),
        console=None,
        command="run_eye_masks_batch --refine",
        created_at_utc=_utc_now(),
        area_filter_z=params.get("area_filter_z", 2.0),
        area_filter_mode=params.get("area_filter_mode", "either"),
        force_refine_traditional=bool(params.get("force_refine_traditional", False)),
        allow_latest_keypoint_fallback=bool(params.get("allow_latest_keypoint_fallback", False)),
    )


def _resolve_registry_model_for_plan(
    *,
    plan: EyeMaskPlan,
    registry_path: Path,
    set_id_filter: Optional[str],
    require_unique: bool,
    top_k: int,
    include_non_success: bool,
    method: str,
    config: Dict[str, Any],
) -> ResolvedModel:
    params = config.get("eye_masks", {}) or {}
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
            task="eye_masks",
            set_id_filter=set_id_filter,
            include_non_success=include_non_success,
        )
    finally:
        registry.close()

    best = _pick_eye_mask_candidate(candidates, require_unique=require_unique)
    payload_args = argparse.Namespace(
        set_id=set_id_filter,
        require_unique=bool(require_unique),
        top_k=int(top_k),
        include_non_success=bool(include_non_success),
        dry_run=False,
        method=method,
        run_name=params.get("run_name"),
        crop_run=params.get("crop_run"),
        keypoints_run=params.get("keypoints_run"),
        batch_size=params.get("batch_size"),
        device=params.get("device"),
        imgsz=params.get("imgsz"),
        conf=params.get("conf"),
        iou=params.get("iou"),
        max_det=params.get("max_det"),
        mask_threshold=params.get("mask_threshold"),
        adaptive_scale=params.get("adaptive_scale"),
        adaptive_cap=params.get("adaptive_cap"),
        no_retina_masks=(params.get("use_retina_masks") is False),
        proto_upsample_factor=params.get("proto_upsample_factor"),
        legacy_masks=bool(params.get("legacy_masks")),
        verbose=bool(params.get("verbose")),
        roi_cache_policy=params.get("roi_cache_policy"),
        roi_cache_dir=params.get("roi_cache_dir"),
        label_mode=params.get("label_mode"),
        write_binary_masks=bool(params.get("write_binary_masks")),
        no_use_crop=not bool(params.get("use_crop", True)),
        json=False,
    )
    payload = _build_eye_mask_resolution_payload(
        args=payload_args,
        argv=None,
        recording_dir=plan.recording_dir,
        output_path=plan.zarr_path,
        registry_path=registry_path,
        recording_id=recording_id,
        target=target,
        selected=best,
        candidates=candidates,
        top_k=int(top_k),
        method=str(method),
    )
    return ResolvedModel(model_path=str(best.model_path), payload=payload)


def _resolve_registry_models_for_plans(
    *,
    plans: Sequence[EyeMaskPlan],
    registry_path: Path,
    set_id_filter: Optional[str],
    require_unique: bool,
    top_k: int,
    include_non_success: bool,
    method: str,
    config: Dict[str, Any],
    on_event: Optional[Callable[[dict[str, object]], None]] = None,
) -> tuple[dict[str, ResolvedModel], dict[str, str]]:
    def _resolve(plan: EyeMaskPlan) -> ResolvedModel:
        return _resolve_registry_model_for_plan(
            plan=plan,
            registry_path=registry_path,
            set_id_filter=set_id_filter,
            require_unique=require_unique,
            top_k=top_k,
            include_non_success=include_non_success,
            method=method,
            config=config,
        )

    return resolve_shared_registry_models_for_plans(
        plans=plans,
        resolve_plan=_resolve,
        on_event=on_event,
    )


def _run_plan(
    plan: EyeMaskPlan,
    *,
    config: Dict[str, Any],
    method: str,
    scheduler: Optional[str],
    num_workers: Optional[int],
    quiet: bool,
    refine: bool,
    refine_only: bool,
    registry_path_for_sync: Optional[Path] = None,
) -> Dict[str, Any]:
    canonical_method = _canonical_method(method)
    payload: Dict[str, Any] = {
        "recording": str(plan.recording_dir),
        "zarr": str(plan.zarr_path),
        "camera_id": plan.camera_id,
        "method": canonical_method,
        "status": "ok",
    }
    stage_start = time.perf_counter()
    stage_payloads: Dict[str, Dict[str, Any]] = {}
    if refine_only:
        source_run = _latest_run(plan.zarr_path, "eye_masks_runs")
        refined_run = _run_refine(plan.zarr_path, config, source_run=source_run, quiet=quiet)
        payload["refined_eye_masks"] = _collect_stage_payload(plan.zarr_path, "refined_eye_masks_runs", refined_run)
        stage_payloads["refined_eye_masks"] = payload["refined_eye_masks"]
    else:
        if canonical_method == "yolo":
            eye_run = _run_yolo(plan.zarr_path, config, quiet=quiet)
        elif canonical_method == "unet":
            eye_run = _run_unet(plan.zarr_path, config)
        else:
            eye_run = _run_traditional(
                plan.zarr_path,
                config,
                scheduler=scheduler,
                num_workers=num_workers,
                quiet=quiet,
            )
        payload["eye_masks"] = _collect_stage_payload(plan.zarr_path, "eye_masks_runs", eye_run)
        stage_payloads["eye_masks"] = payload["eye_masks"]
        if refine:
            refined_run = _run_refine(plan.zarr_path, config, source_run=eye_run, quiet=quiet)
            payload["refined_eye_masks"] = _collect_stage_payload(plan.zarr_path, "refined_eye_masks_runs", refined_run)
            stage_payloads["refined_eye_masks"] = payload["refined_eye_masks"]

    payload["registry_sync"] = _sync_eye_mask_registry_rows_after_run(
        zarr_path=plan.zarr_path,
        stage_payloads=stage_payloads,
        registry_path=registry_path_for_sync,
    )

    payload["duration_seconds"] = time.perf_counter() - stage_start
    return payload


def _run_plan_with_registry_model(
    plan: EyeMaskPlan,
    *,
    config: Dict[str, Any],
    method: str,
    quiet: bool,
    refine: bool,
    registry_path: Path,
    model_set_id: Optional[str],
    model_require_unique: bool,
    model_top_k: int,
    model_include_non_success: bool,
    registry_path_for_sync: Optional[Path] = None,
    resolved_model: Optional[ResolvedModel] = None,
) -> Dict[str, Any]:
    canonical_method = _canonical_method(method)
    if canonical_method not in {"yolo", "unet"}:
        raise ValueError(
            f"Registry model source requires model-based eye mask method; got '{canonical_method}'."
        )

    stage_start = time.perf_counter()
    params = config.get("eye_masks", {}) or {}
    if resolved_model is not None:
        if canonical_method == "unet":
            eye_run = _run_unet(
                plan.zarr_path,
                config,
                checkpoint_override=resolved_model.model_path,
            )
        else:
            eye_run = _run_yolo(
                plan.zarr_path,
                config,
                quiet=quiet,
                model_path_override=resolved_model.model_path,
            )
        _write_eye_mask_model_resolution_provenance(
            zarr_path=plan.zarr_path,
            run_name=eye_run,
            payload=resolved_model.payload,
        )
    else:
        argv: List[str] = [
            "--recording-dir",
            str(plan.recording_dir),
            "--output",
            str(plan.zarr_path),
            "--registry",
            str(registry_path),
            "--method",
            canonical_method,
            "--top-k",
            str(int(model_top_k)),
        ]
        if model_set_id:
            argv.extend(["--set-id", str(model_set_id)])
        if model_require_unique:
            argv.append("--require-unique")
        if model_include_non_success:
            argv.append("--include-non-success")
        if params.get("run_name"):
            argv.extend(["--run-name", str(params.get("run_name"))])
        if params.get("crop_run"):
            argv.extend(["--crop-run", str(params.get("crop_run"))])
        if params.get("keypoints_run"):
            argv.extend(["--keypoints-run", str(params.get("keypoints_run"))])
        if params.get("batch_size") is not None:
            argv.extend(["--batch-size", str(params.get("batch_size"))])
        if params.get("device"):
            argv.extend(["--device", str(params.get("device"))])

        if canonical_method == "yolo":
            if params.get("imgsz") is not None:
                argv.extend(["--imgsz", str(params.get("imgsz"))])
            if params.get("conf") is not None:
                argv.extend(["--conf", str(params.get("conf"))])
            if params.get("iou") is not None:
                argv.extend(["--iou", str(params.get("iou"))])
            if params.get("max_det") is not None:
                argv.extend(["--max-det", str(params.get("max_det"))])
            if params.get("mask_threshold") is not None:
                argv.extend(["--mask-threshold", str(params.get("mask_threshold"))])
            if params.get("adaptive_scale") is not None:
                argv.extend(["--adaptive-scale", str(params.get("adaptive_scale"))])
            if params.get("adaptive_cap") is not None:
                argv.extend(["--adaptive-cap", str(params.get("adaptive_cap"))])
            if params.get("use_retina_masks") is False:
                argv.append("--no-retina-masks")
            if params.get("proto_upsample_factor") is not None:
                argv.extend(["--proto-upsample-factor", str(params.get("proto_upsample_factor"))])
            if params.get("legacy_masks"):
                argv.append("--legacy-masks")
            if params.get("roi_cache_policy"):
                argv.extend(["--roi-cache-policy", str(params.get("roi_cache_policy"))])
            if params.get("roi_cache_dir"):
                argv.extend(["--roi-cache-dir", str(params.get("roi_cache_dir"))])
            if params.get("verbose"):
                argv.append("--verbose")
        else:
            if params.get("label_mode"):
                argv.extend(["--label-mode", str(params.get("label_mode"))])
            if params.get("write_binary_masks"):
                argv.append("--write-binary-masks")
            if params.get("roi_cache_policy"):
                argv.extend(["--roi-cache-policy", str(params.get("roi_cache_policy"))])
            if params.get("roi_cache_dir"):
                argv.extend(["--roi-cache-dir", str(params.get("roi_cache_dir"))])
            if not bool(params.get("use_crop", True)):
                argv.append("--no-use-crop")

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            rc = run_eye_masks_with_registry_model_main(argv)
        if rc != 0:
            stderr_text = stderr_capture.getvalue().strip()
            stdout_text = stdout_capture.getvalue().strip()
            detail = stderr_text or stdout_text or f"exit_code={rc}"
            raise RuntimeError(f"registry eye-mask runner failed: {detail}")

        eye_run = _latest_run(plan.zarr_path, "eye_masks_runs")
        if not eye_run:
            raise RuntimeError("registry eye-mask runner completed but no latest eye_masks run found.")

    payload: Dict[str, Any] = {
        "recording": str(plan.recording_dir),
        "zarr": str(plan.zarr_path),
        "camera_id": plan.camera_id,
        "method": canonical_method,
        "status": "ok",
    }
    stage_payloads: Dict[str, Dict[str, Any]] = {}
    payload["eye_masks"] = _collect_stage_payload(plan.zarr_path, "eye_masks_runs", eye_run)
    stage_payloads["eye_masks"] = payload["eye_masks"]
    if resolved_model is not None:
        selected = (
            resolved_model.payload.get("selected", {})
            if isinstance(resolved_model.payload.get("selected"), dict)
            else {}
        )
        payload["model_resolution"] = {
            "mode": "registry",
            "selected_model_path": selected.get("model_path"),
            "selected_run_id": selected.get("run_id"),
            "selected_set_id": selected.get("set_id"),
            "selected_score": selected.get("score"),
        }

    if refine:
        refined_run = _run_refine(plan.zarr_path, config, source_run=eye_run, quiet=quiet)
        payload["refined_eye_masks"] = _collect_stage_payload(plan.zarr_path, "refined_eye_masks_runs", refined_run)
        stage_payloads["refined_eye_masks"] = payload["refined_eye_masks"]

    payload["registry_sync"] = _sync_eye_mask_registry_rows_after_run(
        zarr_path=plan.zarr_path,
        stage_payloads=stage_payloads,
        registry_path=registry_path_for_sync,
    )
    payload["duration_seconds"] = time.perf_counter() - stage_start
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch run eye-mask segmentation and optional refinement.")
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
    parser.add_argument("--recursive", action="store_true", help="Recursively scan roots.")
    add_apply_dry_run_args(
        parser,
        apply_help="Run eye-mask stages (default: dry-run).",
        dry_run_help="Show plans only (default behavior).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Run segmentation even when eye masks exist.")
    parser.add_argument("--require-crop", action="store_true", help="Require crop_runs latest to exist (default).")
    parser.add_argument("--no-require-crop", action="store_true", help="Allow run without crops.")
    parser.add_argument("--require-keypoints", action="store_true", help="Require keypoint runs to exist (default).")
    parser.add_argument("--no-require-keypoints", action="store_true", help="Allow run without keypoints.")
    parser.add_argument("--config", type=str, default="configs/fisheye/default.yaml", help="Pipeline config path.")
    parser.add_argument("--method", choices=["traditional", "yolo", "unet"], help="Override eye-mask method.")
    parser.add_argument(
        "--scheduler",
        choices=["threads", "processes", "single-threaded", "distributed"],
        default=None,
        help="Scheduler override for traditional eye masks.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Worker count override.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose per-run output.")
    parser.add_argument("--refine", action="store_true", help="Run refine_eye_masks after segmentation.")
    parser.add_argument("--refine-only", action="store_true", help="Run refine_eye_masks only on existing eye masks.")
    parser.add_argument(
        "--refine-missing-only",
        action="store_true",
        help="With --refine-only, skip zarrs that already have refined_eye_masks_runs latest.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON line result payloads.")
    parser.add_argument(
        "--zarr-use",
        choices=["all", "analysis", "training"],
        default="all",
        help="Restrict target zarr scope when scanning paths/file-lists (default: all).",
    )
    add_log_args(
        parser,
        log_dir_help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/run_eye_masks_batch or <recordings_root>/logs/run_eye_masks_batch).",
    )
    # --- Registry discovery mode ---
    add_registry_discovery_args(
        parser,
        camera_flag="--camera-id-filter",
        camera_dest="camera_id_filter",
        camera_help="Filter by camera_id in registry (registry mode). Named --camera-id-filter to avoid ambiguity with per-recording camera_id.",
    )
    parser.add_argument(
        "--model-source",
        choices=["config", "registry"],
        default="config",
        help="Model resolution source for YOLO/U-Net eye masks: config (default) or registry.",
    )
    parser.add_argument(
        "--model-set-id",
        type=str,
        help="Optional set_id filter when --model-source registry.",
    )
    parser.add_argument(
        "--model-require-unique",
        action="store_true",
        help="When --model-source registry, fail plan resolution on top-score ties.",
    )
    parser.add_argument(
        "--model-top-k",
        type=int,
        default=5,
        help="Number of registry model candidates to capture in provenance payloads.",
    )
    parser.add_argument(
        "--model-include-non-success",
        action="store_true",
        help="Include non-success training runs when resolving models from registry.",
    )
    parser.set_defaults(require_crop=True, require_keypoints=True)

    args = parser.parse_args(argv)
    if args.refine and args.refine_only:
        raise SystemExit("Use either --refine or --refine-only, not both.")
    if args.refine_missing_only and not args.refine_only:
        raise SystemExit("Use --refine-missing-only only with --refine-only.")

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
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, log_roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_eye_masks_batch_{run_id}.jsonl"
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
            require_crop=bool(args.require_crop) and not bool(args.no_require_crop),
            require_keypoints=bool(args.require_keypoints) and not bool(args.no_require_keypoints),
            config=args.config,
            method=args.method,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            quiet=bool(args.quiet),
            refine=bool(args.refine),
            refine_only=bool(args.refine_only),
            refine_missing_only=bool(args.refine_missing_only),
            zarr_use=args.zarr_use,
            json=bool(args.json),
            source=args.source,
            sql_prefilter=bool(args.source == "registry" and skip_existing),
            model_source=args.model_source,
            model_set_id=args.model_set_id,
            model_require_unique=bool(args.model_require_unique),
            model_top_k=int(args.model_top_k),
            model_include_non_success=bool(args.model_include_non_success),
        )

    resolved_registry_path: Optional[Path] = None
    if args.source == "registry" or args.model_source == "registry":
        resolved_registry_path = (
            args.registry.expanduser().resolve()
            if args.registry
            else RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
        )

    config = _load_config(args.config)
    try:
        method = _resolve_method(config, args.method)
        if args.model_source == "config":
            _validate_method_requirements(config, method, refine_only=bool(args.refine_only))
        elif not args.refine_only and method not in {"yolo", "unet"}:
            raise ValueError(
                f"--model-source registry requires model-based eye mask method, got '{method}'."
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    require_crop = bool(args.require_crop) and not bool(args.no_require_crop)
    require_keypoints = bool(args.require_keypoints) and not bool(args.no_require_keypoints)

    plans: List[EyeMaskPlan] = []
    if args.source == "registry":
        if resolved_registry_path is None or not resolved_registry_path.exists():
            print(
                f"Error: registry database not found: {resolved_registry_path}",
                file=sys.stderr,
            )
            return 1
        registry_zarrs = _discover_zarrs_from_registry(
            registry_path=resolved_registry_path,
            scope_paths=roots + explicit_zarrs,
            rig_id=args.rig_id,
            arena_id=args.arena_id,
            camera_id=getattr(args, "camera_id_filter", None),
            path_contains=args.path_contains,
            skip_existing=skip_existing,
        )
        if args.emit_paths:
            for p in registry_zarrs:
                print(p)
            return 0
        plans.extend(
            _build_plans_from_zarr(
                registry_zarrs,
                skip_existing=skip_existing,
                require_crop=require_crop,
                require_keypoints=require_keypoints,
                refine_only=bool(args.refine_only),
                zarr_use=str(args.zarr_use),
                refine_missing_only=bool(args.refine_missing_only),
            )
        )
    else:
        if roots:
            plans.extend(
                _build_plans(
                    roots,
                    args.recursive,
                    skip_existing=skip_existing,
                    require_crop=require_crop,
                    require_keypoints=require_keypoints,
                    refine_only=bool(args.refine_only),
                    zarr_use=str(args.zarr_use),
                    refine_missing_only=bool(args.refine_missing_only),
                )
            )
        if explicit_zarrs:
            plans.extend(
                _build_plans_from_zarr(
                    explicit_zarrs,
                    skip_existing=skip_existing,
                    require_crop=require_crop,
                    require_keypoints=require_keypoints,
                    refine_only=bool(args.refine_only),
                    zarr_use=str(args.zarr_use),
                    refine_missing_only=bool(args.refine_missing_only),
                )
            )

    if plans:
        seen: set[str] = set()
        unique: List[EyeMaskPlan] = []
        for plan in plans:
            key = str(plan.zarr_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(plan)
        plans = unique

    if not args.apply:
        if Console is not None and not args.json:
            console = Console()
            console.rule("[bold yellow]Dry run[/bold yellow]")
            console.print("Add [cyan]--apply[/cyan] to run eye-mask stages.")
        if args.json:
            for plan in plans:
                row = {
                    "recording": str(plan.recording_dir),
                    "zarr": str(plan.zarr_path),
                    "camera_id": plan.camera_id,
                    "status": plan.status,
                    "reason": plan.reason,
                }
                print(json.dumps(row, sort_keys=True))
                if logger is not None:
                    logger.log("eye_masks_plan", **row)
        else:
            _print_plan(plans)
            if logger is not None:
                for plan in plans:
                    logger.log(
                        "eye_masks_plan",
                        recording=str(plan.recording_dir),
                        zarr=str(plan.zarr_path),
                        camera_id=plan.camera_id,
                        status=plan.status,
                        reason=plan.reason,
                        eye_masks_present=plan.eye_masks_present,
                        refined_eye_masks_present=plan.refined_eye_masks_present,
                        crop_present=plan.crop_present,
                        keypoints_present=plan.keypoints_present,
                    )
        if logger is not None:
            logger.log("run_end", ok=0, failed=0, skipped=0, missing=0, dry_run=True)
            logger.close()
        return 0

    ok = 0
    failed = 0
    skipped = 0
    missing = 0

    runnable_plans: List[EyeMaskPlan] = []
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            row = {"status": "missing", "zarr": str(plan.zarr_path), "reason": plan.reason}
            if args.json:
                print(json.dumps(row, sort_keys=True))
            else:
                print(f"Skipping (missing prerequisites): {plan.zarr_path} ({plan.reason})")
            if logger is not None:
                logger.log("eye_masks_skipped", **row)
            continue
        if plan.status == "skipped" and (
            skip_existing or plan.reason == "refined eye masks already present"
        ):
            skipped += 1
            row = {"status": "skipped", "zarr": str(plan.zarr_path), "reason": plan.reason or "skipped"}
            if args.json:
                print(json.dumps(row, sort_keys=True))
            else:
                print(f"Skipping ({row['reason']}): {plan.zarr_path}")
            if logger is not None:
                logger.log("eye_masks_skipped", **row)
            continue
        runnable_plans.append(plan)

    resolved_models_by_plan: dict[str, ResolvedModel] = {}
    if args.model_source == "registry" and not args.refine_only and runnable_plans:
        if resolved_registry_path is None or not resolved_registry_path.exists():
            print(
                f"Error: registry database not found for model resolution: {resolved_registry_path}",
                file=sys.stderr,
            )
            return 1
        if not args.json and not args.quiet:
            print(f"Pre-resolving registry models for {len(runnable_plans)} recording plan(s)...")

        def _emit_model_resolution_event(event_payload: dict[str, object]) -> None:
            event_name = str(event_payload.get("event", "model_resolution_event"))
            if logger is not None:
                logger.log(
                    event_name,
                    **{k: v for k, v in event_payload.items() if k != "event"},
                )
            if args.json or args.quiet:
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
            registry_path=resolved_registry_path,
            set_id_filter=args.model_set_id,
            require_unique=bool(args.model_require_unique),
            top_k=int(args.model_top_k),
            include_non_success=bool(args.model_include_non_success),
            method=method,
            config=config,
            on_event=_emit_model_resolution_event,
        )
        if logger is not None:
            logger.log(
                "model_resolution_summary",
                total=len(runnable_plans),
                resolved=len(resolved_models_by_plan),
                failed=len(resolution_errors),
            )
        if not args.json and not args.quiet:
            print(
                "Model resolution summary: "
                f"resolved={len(resolved_models_by_plan)} failed={len(resolution_errors)}"
            )
        if resolution_errors:
            runnable_after_resolution: List[EyeMaskPlan] = []
            for plan in runnable_plans:
                key = str(plan.zarr_path.resolve())
                error = resolution_errors.get(key)
                if error is None:
                    runnable_after_resolution.append(plan)
                    continue
                failed += 1
                row = {
                    "status": "failed",
                    "zarr": str(plan.zarr_path),
                    "error": f"model resolution failed: {error}",
                }
                if args.json:
                    print(json.dumps(row, sort_keys=True))
                else:
                    print(f"Eye masks failed for {plan.zarr_path}: model resolution failed: {error}")
                if logger is not None:
                    logger.log("eye_masks_failed", **row)
            runnable_plans = runnable_after_resolution

    console = Console() if (Console is not None and not args.json and not args.quiet) else None
    progress = None
    quiet = bool(args.quiet or args.json)
    task_label = "Refining eye masks" if args.refine_only else "Running eye masks"

    def _handle_plan(plan: EyeMaskPlan) -> None:
        nonlocal ok, failed
        try:
            if args.model_source == "registry" and not args.refine_only:
                result = _run_plan_with_registry_model(
                    plan,
                    config=config,
                    method=method,
                    quiet=quiet,
                    refine=bool(args.refine),
                    registry_path=resolved_registry_path,
                    model_set_id=args.model_set_id,
                    model_require_unique=bool(args.model_require_unique),
                    model_top_k=int(args.model_top_k),
                    model_include_non_success=bool(args.model_include_non_success),
                    registry_path_for_sync=resolved_registry_path,
                    resolved_model=resolved_models_by_plan.get(str(plan.zarr_path.resolve())),
                )
            else:
                result = _run_plan(
                    plan,
                    config=config,
                    method=method,
                    scheduler=args.scheduler,
                    num_workers=args.num_workers,
                    quiet=quiet,
                    refine=bool(args.refine),
                    refine_only=bool(args.refine_only),
                    registry_path_for_sync=resolved_registry_path,
                )
            sync_payload = result.get("registry_sync")
            if (
                isinstance(sync_payload, dict)
                and not bool(sync_payload.get("synced"))
                and not args.quiet
                and not args.json
            ):
                reason = sync_payload.get("reason") or "registry_sync_failed"
                error = sync_payload.get("error")
                if error:
                    print(f"Warning: registry sync incomplete for {plan.zarr_path} ({reason}: {error})")
                else:
                    print(f"Warning: registry sync incomplete for {plan.zarr_path} ({reason})")
            ok += 1
            if args.json:
                print(json.dumps(result, sort_keys=True))
            if logger is not None:
                logger.log("eye_masks_ok", **result)
        except Exception as exc:
            failed += 1
            row = {
                "status": "failed",
                "recording": str(plan.recording_dir),
                "zarr": str(plan.zarr_path),
                "method": method,
                "failure_reason": str(exc),
            }
            if args.json:
                print(json.dumps(row, sort_keys=True))
            else:
                print(f"Eye masks failed for {plan.zarr_path}: {exc}")
            if logger is not None:
                logger.log("eye_masks_failed", **row)

    if progress is None:
        total_plans = len(runnable_plans)
        for idx, plan in enumerate(runnable_plans, start=1):
            if console is not None:
                console.print(
                    f"[bold cyan][{idx}/{total_plans}] {task_label}[/bold cyan] "
                    f"[dim]{plan.recording_dir}[/dim]"
                )
            _handle_plan(plan)
    else:
        with progress:
            task = progress.add_task(task_label, total=len(runnable_plans))
            for plan in runnable_plans:
                _handle_plan(plan)
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
