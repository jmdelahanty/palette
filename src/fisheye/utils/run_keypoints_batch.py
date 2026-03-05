import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np
import yaml
import zarr

from fisheye.detection.detect_keypoints_traditional import detect_keypoints
from fisheye.detection.detect_keypoints_yolo import detect_keypoints_yolo
from fisheye.refinement.refine_keypoints import create_refined_keypoint_run
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.status_ledger import upsert_recording_step_status
from fisheye.utils.auto_keypoint_review import AutoReviewPolicy, apply_auto_review
from fisheye.utils.model_resolution_provenance import build_model_resolution_payload
from fisheye.utils import refine_keypoints_batch as refine_keypoints_batch_mod
from fisheye.utils.resolve_detect_model import Candidate
from fisheye.utils.resolve_detect_model import _load_candidates, _load_target_profile, _resolve_recording_id
from fisheye.utils.run_keypoints_with_registry_model import (
    _write_model_resolution_provenance as _write_keypoint_model_resolution_provenance,
)

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
    h5_path: Optional[Path]
    zarr_path: Path
    camera_id: Optional[str]
    status: str
    reason: Optional[str] = None
    keypoints_present: bool = False
    crop_present: bool = False
    background_present: bool = False
    tuning_present: bool = False


@dataclass(frozen=True)
class ResolvedModel:
    model_path: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class RegistryZarrEntry:
    zarr_path: Path
    camera_id: Optional[str]


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


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


def _latest_run(zarr_path: Path, group_name: str) -> Optional[str]:
    try:
        root = zarr.open(str(zarr_path), mode="r", use_consolidated=False)
    except Exception:
        return None
    group = root.get(group_name)
    if group is None:
        return None
    latest = group.attrs.get("latest")
    return str(latest) if latest else None


def _latest_keypoints_run(zarr_path: Path) -> Optional[str]:
    return _latest_run(zarr_path, "keypoints_runs")


def _keypoints_total_rois(zarr_path: Path, run_name: Optional[str]) -> Optional[int]:
    try:
        root = zarr.open(str(zarr_path), mode="r", use_consolidated=False)
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


def _as_nonnegative_int(value: object) -> Optional[int]:
    try:
        int_value = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return int_value if int_value >= 0 else None


def _hash_parameters(params: object) -> Optional[str]:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        payload = str(params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_keypoint_signature(attrs: Dict[str, Any]) -> Dict[str, object]:
    params = attrs.get("parameters")
    if not isinstance(params, dict):
        provenance = attrs.get("provenance")
        if isinstance(provenance, dict):
            params = provenance.get("parameters")
    if not isinstance(params, dict):
        params = None

    parameter_source = attrs.get("parameter_source")
    if parameter_source is None and isinstance(params, dict):
        parameter_source = params.get("parameter_source")

    return {
        "signature_version": 1,
        "source_keypoints_run": attrs.get("source_keypoints_run"),
        "source_crop_run": attrs.get("source_crop_run"),
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "parameter_source": parameter_source,
        "parameters_hash": _hash_parameters(params),
    }


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


def _refined_usable_counts(refined: zarr.Group) -> tuple[Optional[int], Optional[int]]:
    total: Optional[int] = None
    usable: Optional[int] = None

    summary = refined.attrs.get("summary_statistics")
    if isinstance(summary, dict):
        total = _as_nonnegative_int(summary.get("total_rois"))
        usable = _as_nonnegative_int(summary.get("usable_keypoints"))

    if (total is None or usable is None) and "usable_keypoints" in refined:
        try:
            usable_values = np.asarray(refined["usable_keypoints"][:], dtype=bool)
            total = int(usable_values.shape[0])
            usable = int(np.count_nonzero(usable_values))
        except Exception:
            pass

    return total, usable


def _maybe_auto_approve_refined_keypoints(
    *,
    zarr_path: Path,
    refined_run: str,
    min_usable_rate: float,
    intended_use: str,
    reviewer: Optional[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "threshold": float(min_usable_rate),
        "intended_use": intended_use,
        "refined_run": refined_run,
    }

    policy = AutoReviewPolicy(
        policy_id="keypoint_auto_review_v1",
        policy_version=1,
        remaining_failures_max=10**12,
        refined_success_rate_min=0.0,
        usable_keypoints_rate_min=float(min_usable_rate),
        confidence_valid_rate_min=0.0,
        geometry_valid_rate_min=0.0,
        disqualifying_tags=(),
    )
    try:
        dry_result = apply_auto_review(
            zarr_path,
            refined_run=refined_run,
            source_keypoints_run=None,
            policy=policy,
            intended_use=intended_use,
            reviewer=reviewer,
            state_on_pass="approved",
            state_on_fail="needs_review",
            overwrite_existing=False,
            update_latest=True,
            dry_run=True,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "No refined_keypoints_runs found" in message:
            result["reason"] = "refined_keypoints_parent_missing"
            return result
        if "Refined keypoint run" in message and "not found" in message:
            result["reason"] = "refined_run_missing"
            return result
        result["reason"] = "auto_review_error"
        result["error"] = message
        return result

    payload = dry_result.get("payload")
    if isinstance(payload, dict):
        auto_review = payload.get("auto_review")
        if isinstance(auto_review, dict):
            evidence = auto_review.get("evidence")
            if isinstance(evidence, dict):
                result["total_rois"] = evidence.get("total_rois")
                result["usable_keypoints"] = evidence.get("usable_keypoints")
                result["usable_rate"] = evidence.get("usable_keypoints_rate")

    if dry_result.get("skipped"):
        skip_reason = str(dry_result.get("skip_reason") or "skipped")
        if skip_reason == "existing_review_status":
            result["reason"] = "already_reviewed"
            existing_payload = dry_result.get("payload")
            if isinstance(existing_payload, dict):
                result["existing_review_status"] = _jsonable(existing_payload)
            return result
        result["reason"] = skip_reason
        return result

    if not bool(dry_result.get("passed")):
        result["reason"] = "threshold_not_met"
        result["evaluation"] = _jsonable(payload)
        return result

    notes = (
        "Auto-approved by run_keypoints_batch: "
        f"usable_keypoints_rate >= threshold={float(min_usable_rate):.6f}"
    )
    write_result = apply_auto_review(
        zarr_path,
        refined_run=refined_run,
        source_keypoints_run=None,
        policy=policy,
        intended_use=intended_use,
        reviewer=reviewer,
        notes=notes,
        state_on_pass="approved",
        state_on_fail="needs_review",
        overwrite_existing=False,
        update_latest=True,
        dry_run=False,
    )
    result["applied"] = not bool(write_result.get("skipped"))
    result["reason"] = "threshold_met" if result["applied"] else str(write_result.get("skip_reason") or "skipped")
    write_payload = write_result.get("payload")
    if isinstance(write_payload, dict):
        result["review_status"] = _jsonable(write_payload)
    return result


def _parse_json_mapping(value: object) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return dict(parsed)
    return None


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


def _coverage_pct_from_stage_payload(stage_payload: Dict[str, Any]) -> Optional[float]:
    summary = stage_payload.get("summary_statistics")
    if isinstance(summary, dict):
        for key in ("pass_rate_percent", "success_rate_percent"):
            if key in summary:
                try:
                    return float(summary[key])
                except (TypeError, ValueError):
                    continue
    success_rate = stage_payload.get("success_rate")
    if success_rate is not None:
        try:
            value = float(success_rate)
            if value <= 1.0:
                value *= 100.0
            return value
        except (TypeError, ValueError):
            return None
    return None


def _zarr_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return None


def _sync_refined_keypoint_step_status_after_auto_review(
    *,
    zarr_path: Path,
    refined_run: str,
    stage_payload: Dict[str, Any],
    review_status: Dict[str, Any],
) -> Dict[str, Any]:
    context = _resolve_step_status_context(zarr_path)
    if context is None:
        return {"synced": False, "reason": "status_context_unavailable"}

    details_update: Dict[str, Any] = {
        "reason": "present",
        "latest_selector": "run_keypoints_batch_auto_review",
    }
    source_runs = stage_payload.get("source_runs")
    if isinstance(source_runs, dict):
        for key in ("source_keypoints_run", "source_crop_run", "source_detect_run"):
            if source_runs.get(key) is not None:
                details_update[key] = source_runs.get(key)

    summary = stage_payload.get("summary_statistics")
    if isinstance(summary, dict):
        for key in ("total_rois", "refined_success", "usable_keypoints", "pass_rate_percent"):
            if summary.get(key) is not None:
                details_update[key] = summary.get(key)

    registry = Registry(Path(str(context["registry_path"])))
    try:
        existing_row = registry.conn.execute(
            """
            SELECT details_json
            FROM recording_step_status
            WHERE dataset_id = ? AND step_name = 'refined_keypoints'
            LIMIT 1;
            """,
            (str(context["dataset_id"]),),
        ).fetchone()
        existing_details = _parse_json_mapping(existing_row["details_json"]) if existing_row is not None else None
        merged_details: Dict[str, Any] = {}
        if isinstance(existing_details, dict):
            merged_details.update(existing_details)
        merged_details.update(details_update)

        upsert_recording_step_status(
            registry,
            dataset_id=str(context["dataset_id"]),
            recording_id=context.get("recording_id"),
            step_name="refined_keypoints",
            status="ok",
            run_name=refined_run,
            method=_normalize_attr(stage_payload.get("method")) or "refine_keypoints",
            coverage_pct=_coverage_pct_from_stage_payload(stage_payload),
            review_status_json=review_status,
            details_json=merged_details,
            source="run_keypoints_batch_auto_review",
            zarr_mtime_ns=_zarr_mtime_ns(zarr_path),
        )
        return {
            "synced": True,
            "dataset_id": str(context["dataset_id"]),
            "recording_id": context.get("recording_id"),
        }
    except Exception as exc:
        return {"synced": False, "reason": "upsert_failed", "error": str(exc)}
    finally:
        registry.close()


def _sync_keypoint_registry_rows_after_run(
    *,
    zarr_path: Path,
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
        perf_rows = int(
            registry.refresh_keypoint_performance_for_dataset(
                dataset_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
            )
        )
        quality_rows = int(registry.refresh_keypoint_quality_for_dataset(dataset_id, zarr_path=zarr_path))
        return {
            "synced": True,
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "keypoint_performance_rows": perf_rows,
            "keypoint_quality_rows": quality_rows,
        }
    except Exception as exc:
        return {"synced": False, "reason": "refresh_failed", "error": str(exc)}
    finally:
        registry.close()


def _decode_source_runs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    source_runs: Dict[str, Any] = {}
    for key in (
        "source_keypoints_run",
        "source_crop_run",
        "source_detect_run",
        "source_refined_run",
    ):
        if key in attrs:
            source_runs[key] = _jsonable(attrs.get(key))
    return source_runs


def _collect_stage_payload(zarr_path: Path, group_name: str, run_name: str) -> Dict[str, Any]:
    root = zarr.open(str(zarr_path), mode="r", use_consolidated=False)
    parent = root.get(group_name)
    if parent is None or run_name not in parent:
        raise RuntimeError(f"run not found after stage completion: {group_name}/{run_name}")
    run_group = parent[run_name]
    attrs = dict(run_group.attrs)

    summary = attrs.get("summary_statistics")
    summary_stats = summary if isinstance(summary, dict) else {}
    duration_seconds = attrs.get("duration_seconds")
    if duration_seconds is None:
        duration_seconds = attrs.get("inference_duration_seconds")
    if duration_seconds is None and isinstance(summary_stats, dict):
        duration_seconds = summary_stats.get("duration_seconds")

    method = _normalize_attr(attrs.get("method"))
    if not method:
        method = _normalize_attr(attrs.get("refinement_role"))
    if not method and group_name == "refined_keypoints_runs":
        method = "refine_keypoints"

    return {
        "group": group_name,
        "run_name": run_name,
        "method": method,
        "duration_seconds": _jsonable(duration_seconds),
        "keypoints_processed": _jsonable(attrs.get("keypoints_processed")),
        "success_rate": _jsonable(attrs.get("success_rate")),
        "summary_statistics": _jsonable(summary_stats),
        "source_runs": _decode_source_runs(attrs),
    }


def _plan_from_zarr(
    *,
    zarr_path: Path,
    recording_dir: Path,
    h5_path: Optional[Path],
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
    entries = _discover_registry_entries(
        registry_path=registry_path,
        scope_paths=scope_paths,
        rig_id=rig_id,
        arena_id=arena_id,
        camera_id=camera_id,
        path_contains=path_contains,
        skip_existing=skip_existing,
    )
    return [entry.zarr_path for entry in entries]


def _discover_registry_entries(
    *,
    registry_path: Path,
    scope_paths: Sequence[Path],
    rig_id: Optional[str] = None,
    arena_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    path_contains: Optional[str] = None,
    skip_existing: bool = False,
) -> List[RegistryZarrEntry]:
    """Query the registry for analysis zarr paths suitable for keypoint processing.

    When *skip_existing* is True, recordings whose ``keypoints`` step status is
    already ``'ok'`` are excluded at the SQL level.  Additionally, only
    recordings whose ``detect`` and ``crop`` steps are both ``'ok'`` are
    returned, ensuring prerequisite pipeline stages are complete.
    """
    registry = Registry(registry_path)
    try:
        query_kwargs: dict[str, Any] = dict(
            zarr_use="analysis",
            exclude_status="missing",
            require_recording=True,
            require_steps_ok=["detect", "crop"],
            rig_id=rig_id,
            arena_id=arena_id,
            camera_id=camera_id,
            path_contains=path_contains,
        )
        if skip_existing:
            query_kwargs["exclude_step_ok"] = "keypoints"
        rows = registry.query_datasets(**query_kwargs)
    finally:
        registry.close()

    entries: list[RegistryZarrEntry] = []
    for row in rows:
        raw = row["zarr_path"]
        if raw is None:
            continue
        zarr_path = Path(str(raw))
        if not zarr_path.name.endswith("_analysis.zarr"):
            continue
        entries.append(
            RegistryZarrEntry(
                zarr_path=zarr_path.resolve(),
                camera_id=_normalize_attr(row["camera_id"]),
            )
        )

    # Scope filter: if caller provided paths, keep only zarrs under those roots.
    if scope_paths:
        resolved_scopes = [str(p.expanduser().resolve()).rstrip("/") + "/" for p in scope_paths]
        entries = [entry for entry in entries if any(str(entry.zarr_path).startswith(s) for s in resolved_scopes)]

    # Deduplicate and sort by zarr path (match filesystem discovery contract).
    merged: Dict[str, RegistryZarrEntry] = {}
    for entry in entries:
        key = str(entry.zarr_path)
        existing = merged.get(key)
        if existing is None:
            merged[key] = entry
            continue
        if existing.camera_id is None and entry.camera_id is not None:
            merged[key] = entry
    ordered = sorted(merged.values(), key=lambda item: str(item.zarr_path))
    return ordered


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


def _resolve_registry_path(path: Optional[Path]) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    return RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()


def _pick_best_candidate(candidates: List[Candidate], *, require_unique: bool) -> Candidate:
    if not candidates:
        raise RuntimeError("No pose model candidates found.")
    best = candidates[0]
    if require_unique and len(candidates) > 1:
        if abs(candidates[0].weighted_score - candidates[1].weighted_score) < 1e-12:
            raise RuntimeError("Top candidate score tied; rerun with --model-set-id to choose deterministically.")
    return best


def _candidate_payload(item: Candidate) -> Dict[str, Any]:
    return {
        "run_id": item.run_id,
        "set_id": item.set_id,
        "model_path": item.model_path,
        "score": item.weighted_score,
        "created_utc": item.created_utc,
        "status": item.status,
        "dataset_count": item.dataset_count,
        "feature_match_counts": item.feature_match_counts,
        "feature_weights_used": item.feature_weights_used,
    }


def _resolve_registry_model_for_plan(
    *,
    plan: KeypointPlan,
    registry_path: Path,
    set_id_filter: Optional[str],
    require_unique: bool,
    top_k: int,
    include_non_success: bool,
    crop_run: Optional[str],
    batch_size: Optional[int],
    device: Optional[str],
    imgsz: Optional[int],
    conf: Optional[float],
    iou: Optional[float],
    max_det: Optional[int],
    mask_threshold: Optional[float],
) -> ResolvedModel:
    registry = Registry(registry_path)
    try:
        recording_id = _resolve_recording_id(
            registry,
            recording_id=None,
            recording_dir=plan.recording_dir,
        )
        target = _load_target_profile(registry, recording_id)
        candidates = _load_candidates(
            registry,
            target=target,
            task="pose",
            set_id_filter=set_id_filter,
            include_non_success=include_non_success,
        )
    finally:
        registry.close()

    best = _pick_best_candidate(candidates, require_unique=require_unique)
    selected_payload = _candidate_payload(best)
    candidate_payloads = [_candidate_payload(item) for item in candidates[: max(0, int(top_k))]]
    invocation_args = argparse.Namespace(
        model_source="registry",
        model_set_id=set_id_filter,
        model_require_unique=bool(require_unique),
        model_top_k=int(top_k),
        model_include_non_success=bool(include_non_success),
        crop_run=crop_run,
        batch_size=batch_size,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        mask_threshold=mask_threshold,
    )
    payload = build_model_resolution_payload(
        tool="fisheye.utils.run_keypoints_batch",
        args=invocation_args,
        argv=None,
        task="pose",
        registry_path=registry_path,
        recording_id=recording_id,
        target=asdict(target),
        selected=selected_payload,
        candidates=candidate_payloads,
        parameters={
            "set_id_filter": set_id_filter,
            "require_unique": bool(require_unique),
            "top_k": int(top_k),
            "include_non_success": bool(include_non_success),
        },
        inputs={
            "recording_dir": str(plan.recording_dir),
            "output_zarr": str(plan.zarr_path),
            "recording_id": recording_id,
        },
        artifacts={
            "selected_model": selected_payload,
            "candidate_models": candidate_payloads,
            "output_zarr": str(plan.zarr_path),
        },
    )
    return ResolvedModel(model_path=str(best.model_path), payload=payload)


def _resolve_registry_models_for_plans(
    *,
    plans: List[KeypointPlan],
    registry_path: Path,
    set_id_filter: Optional[str],
    require_unique: bool,
    top_k: int,
    include_non_success: bool,
    config: Dict[str, Any],
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[Dict[str, ResolvedModel], Dict[str, str]]:
    params = config.get("keypoints", {})
    resolved: Dict[str, ResolvedModel] = {}
    errors: Dict[str, str] = {}
    resolved_by_recording: Dict[str, ResolvedModel] = {}
    errors_by_recording: Dict[str, str] = {}
    total = len(plans)
    for index, plan in enumerate(plans, start=1):
        key = str(plan.zarr_path.resolve())
        recording_key = str(plan.recording_dir.resolve())
        base_event: Dict[str, Any] = {
            "index": index,
            "total": total,
            "recording": str(plan.recording_dir),
            "zarr": str(plan.zarr_path),
            "camera_id": plan.camera_id,
        }
        cached = resolved_by_recording.get(recording_key)
        if cached is not None:
            resolved[key] = cached
            selected = cached.payload.get("selected")
            selected_payload = selected if isinstance(selected, dict) else {}
            if on_event is not None:
                on_event(
                    {
                        **base_event,
                        "event": "model_resolution_cached",
                        "selected_model_path": cached.model_path,
                        "selected_run_id": selected_payload.get("run_id"),
                        "selected_set_id": selected_payload.get("set_id"),
                    }
                )
            continue
        cached_error = errors_by_recording.get(recording_key)
        if cached_error is not None:
            errors[key] = cached_error
            if on_event is not None:
                on_event(
                    {
                        **base_event,
                        "event": "model_resolution_failed",
                        "error": cached_error,
                    }
                )
            continue
        if on_event is not None:
            on_event({**base_event, "event": "model_resolution_start"})
        try:
            model = _resolve_registry_model_for_plan(
                plan=plan,
                registry_path=registry_path,
                set_id_filter=set_id_filter,
                require_unique=require_unique,
                top_k=top_k,
                include_non_success=include_non_success,
                crop_run=params.get("crop_run"),
                batch_size=params.get("batch_size", 256),
                device=params.get("device"),
                imgsz=params.get("imgsz"),
                conf=params.get("conf", 0.25),
                iou=params.get("iou", 0.5),
                max_det=params.get("max_det", 1),
                mask_threshold=params.get("mask_threshold", 0.5),
            )
            resolved[key] = model
            resolved_by_recording[recording_key] = model
            selected = model.payload.get("selected")
            selected_payload = selected if isinstance(selected, dict) else {}
            if on_event is not None:
                on_event(
                    {
                        **base_event,
                        "event": "model_resolution_ok",
                        "selected_model_path": model.model_path,
                        "selected_run_id": selected_payload.get("run_id"),
                        "selected_set_id": selected_payload.get("set_id"),
                    }
                )
        except Exception as exc:
            message = str(exc)
            errors[key] = message
            errors_by_recording[recording_key] = message
            if on_event is not None:
                on_event(
                    {
                        **base_event,
                        "event": "model_resolution_failed",
                        "error": message,
                    }
                )
    return resolved, errors


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


def _run_yolo(
    zarr_path: str,
    config: Dict[str, Any],
    quiet: bool,
    model_path_override: Optional[str] = None,
) -> str:
    params = config.get("keypoints", {})
    model_path = model_path_override or params.get("model") or params.get("model_path")
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


def _run_plan(
    plan: KeypointPlan,
    *,
    config: Dict[str, Any],
    method: str,
    scheduler: Optional[str],
    num_workers: Optional[int],
    quiet: bool,
    dask_progress: bool,
    refine: bool,
    refine_only: bool,
    json_output: bool,
    resolved_model: Optional[ResolvedModel] = None,
    auto_approve_min_usable_rate: Optional[float] = None,
    auto_approve_intended_use: str = "full_recording",
    auto_approve_reviewer: Optional[str] = None,
    registry_path_for_sync: Optional[Path] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "recording": str(plan.recording_dir),
        "zarr": str(plan.zarr_path),
        "camera_id": plan.camera_id,
        "method": method,
        "status": "ok",
    }
    stage_start = time.perf_counter()
    updated_keypoint_outputs = False

    if refine_only:
        source_run = _latest_keypoints_run(plan.zarr_path)
        payload["source_keypoints_run"] = source_run
        total_rois = _keypoints_total_rois(plan.zarr_path, source_run)
        if total_rois is None or total_rois > 0:
            refined_run = _run_refine(str(plan.zarr_path), config, source_run, quiet=quiet)
            payload["refined_keypoints"] = _collect_stage_payload(
                plan.zarr_path,
                "refined_keypoints_runs",
                refined_run,
            )
            updated_keypoint_outputs = True
            if auto_approve_min_usable_rate is not None:
                payload["auto_review"] = _maybe_auto_approve_refined_keypoints(
                    zarr_path=plan.zarr_path,
                    refined_run=refined_run,
                    min_usable_rate=float(auto_approve_min_usable_rate),
                    intended_use=auto_approve_intended_use,
                    reviewer=auto_approve_reviewer,
                )
                auto_review_payload = payload.get("auto_review")
                if isinstance(auto_review_payload, dict):
                    review_status = auto_review_payload.get("review_status")
                    if not isinstance(review_status, dict):
                        review_status = auto_review_payload.get("existing_review_status")
                    if isinstance(review_status, dict):
                        auto_review_payload["step_status_sync"] = _sync_refined_keypoint_step_status_after_auto_review(
                            zarr_path=plan.zarr_path,
                            refined_run=refined_run,
                            stage_payload=payload["refined_keypoints"],
                            review_status=review_status,
                        )
        else:
            payload["refine_skipped_reason"] = "zero_keypoint_rois"
    else:
        if method in {"yolo", "yolo_pose"}:
            run_name = _run_yolo(
                str(plan.zarr_path),
                config,
                quiet=quiet,
                model_path_override=(resolved_model.model_path if resolved_model else None),
            )
        else:
            _run_traditional(
                str(plan.zarr_path),
                config,
                scheduler=scheduler,
                num_workers=num_workers,
                quiet=quiet,
                show_progress=dask_progress and not json_output,
            )
            run_name = _latest_keypoints_run(plan.zarr_path)
            if not run_name:
                raise RuntimeError("keypoints run not found after detection")
        payload["keypoints"] = _collect_stage_payload(
            plan.zarr_path,
            "keypoints_runs",
            run_name,
        )
        updated_keypoint_outputs = True
        if resolved_model is not None:
            _write_keypoint_model_resolution_provenance(
                zarr_path=plan.zarr_path,
                run_name=run_name,
                payload=resolved_model.payload,
            )
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
            total_rois = _keypoints_total_rois(plan.zarr_path, run_name)
            if total_rois is None or total_rois > 0:
                refined_run = _run_refine(str(plan.zarr_path), config, run_name, quiet=quiet)
                payload["refined_keypoints"] = _collect_stage_payload(
                    plan.zarr_path,
                    "refined_keypoints_runs",
                    refined_run,
                )
                updated_keypoint_outputs = True
                if auto_approve_min_usable_rate is not None:
                    payload["auto_review"] = _maybe_auto_approve_refined_keypoints(
                        zarr_path=plan.zarr_path,
                        refined_run=refined_run,
                        min_usable_rate=float(auto_approve_min_usable_rate),
                        intended_use=auto_approve_intended_use,
                        reviewer=auto_approve_reviewer,
                    )
                    auto_review_payload = payload.get("auto_review")
                    if isinstance(auto_review_payload, dict):
                        review_status = auto_review_payload.get("review_status")
                        if not isinstance(review_status, dict):
                            review_status = auto_review_payload.get("existing_review_status")
                        if isinstance(review_status, dict):
                            auto_review_payload["step_status_sync"] = _sync_refined_keypoint_step_status_after_auto_review(
                                zarr_path=plan.zarr_path,
                                refined_run=refined_run,
                                stage_payload=payload["refined_keypoints"],
                                review_status=review_status,
                            )
            else:
                payload["refine_skipped_reason"] = "zero_keypoint_rois"

    if updated_keypoint_outputs:
        payload["registry_sync"] = _sync_keypoint_registry_rows_after_run(
            zarr_path=plan.zarr_path,
            registry_path=registry_path_for_sync,
        )

    payload["duration_seconds"] = time.perf_counter() - stage_start
    return payload


def _delegate_refine_only(args: argparse.Namespace) -> int:
    delegate_argv: List[str] = []
    delegate_argv.extend(str(path) for path in args.paths)
    for file_list in args.file_list or []:
        delegate_argv.extend(["--file-list", str(file_list)])
    if args.recursive:
        delegate_argv.append("--recursive")
    if args.apply:
        delegate_argv.append("--apply")
    # Preserve historical refine-only behavior from run_keypoints_batch:
    # do not skip when refined runs already exist.
    delegate_argv.extend(["--zarr-use", "any", "--no-skip-existing"])
    delegate_argv.extend(["--config", str(args.config)])
    if args.scheduler in {"processes", "threads", "distributed"}:
        delegate_argv.extend(["--scheduler", str(args.scheduler)])
    elif args.scheduler == "single-threaded":
        print(
            "Warning: --scheduler=single-threaded is not supported by refine_keypoints_batch; ignoring.",
            file=sys.stderr,
        )
    if args.num_workers is not None:
        delegate_argv.extend(["--num-workers", str(args.num_workers)])
    if args.log_dir is not None:
        delegate_argv.extend(["--log-dir", str(args.log_dir)])
    if args.no_log:
        delegate_argv.append("--no-log")
    if args.json:
        delegate_argv.append("--json")
    return int(refine_keypoints_batch_mod.main(delegate_argv))


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
        help="Require a background run to exist before running keypoints (default for traditional).",
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
        "--auto-approve-perfect",
        action="store_true",
        help=(
            "After refinement, auto-set keypoint_review_status to approved/algorithmic "
            "when usable keypoint coverage is 100%."
        ),
    )
    parser.add_argument(
        "--auto-approve-min-usable-rate",
        type=float,
        help=(
            "After refinement, auto-set keypoint_review_status to approved/algorithmic "
            "when usable_keypoints/total_rois >= this threshold (0-1)."
        ),
    )
    parser.add_argument(
        "--auto-approve-intended-use",
        choices=["training", "full_recording"],
        default="full_recording",
        help="Review intended_use to write for auto-approvals (default: full_recording).",
    )
    parser.add_argument(
        "--auto-approve-reviewer",
        type=str,
        help="Optional reviewer label to include in auto-approval payloads.",
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
    # --- Registry discovery mode ---
    parser.add_argument(
        "--source",
        choices=["filesystem", "registry"],
        default="filesystem",
        help="Discovery source: filesystem (default) or registry.",
    )
    parser.add_argument(
        "--emit-paths",
        action="store_true",
        help="Print discovered zarr paths (one per line) and exit.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Path to registry SQLite database (required when --source registry).",
    )
    parser.add_argument(
        "--rig-id",
        type=str,
        help="Filter by rig_id (registry mode).",
    )
    parser.add_argument(
        "--arena-id",
        type=str,
        help="Filter by arena_id (registry mode).",
    )
    parser.add_argument(
        "--camera-id-filter",
        type=str,
        help="Filter by camera_id in registry (registry mode). Named --camera-id-filter to avoid ambiguity with per-recording camera_id.",
    )
    parser.add_argument(
        "--path-contains",
        type=str,
        help="Filter zarr_path by substring (registry mode).",
    )
    parser.add_argument(
        "--model-source",
        choices=["config", "registry"],
        default="config",
        help="YOLO model resolution source: config (default) or registry.",
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
    parser.set_defaults(require_crop=True)

    args = parser.parse_args(argv)
    if args.auto_approve_perfect and args.auto_approve_min_usable_rate is not None:
        print(
            "Error: use either --auto-approve-perfect or --auto-approve-min-usable-rate, not both.",
            file=sys.stderr,
        )
        return 1
    auto_approve_min_usable_rate: Optional[float] = None
    if args.auto_approve_perfect:
        auto_approve_min_usable_rate = 1.0
    elif args.auto_approve_min_usable_rate is not None:
        auto_approve_min_usable_rate = float(args.auto_approve_min_usable_rate)
    if auto_approve_min_usable_rate is not None and not (0.0 <= auto_approve_min_usable_rate <= 1.0):
        print("Error: --auto-approve-min-usable-rate must be between 0 and 1.", file=sys.stderr)
        return 1
    if auto_approve_min_usable_rate is not None and not (args.refine or args.refine_only):
        print(
            "Error: auto-approval requires --refine (or legacy --refine-only).",
            file=sys.stderr,
        )
        return 1
    if args.refine_only:
        deprecation_msg = (
            "Deprecation warning: `run_keypoints_batch --refine-only` is deprecated. "
            "Use `scripts/py -m fisheye.utils.refine_keypoints_batch`."
        )
        print(deprecation_msg, file=sys.stderr)
        if auto_approve_min_usable_rate is not None:
            print(
                "Error: auto-approval is not supported with --refine-only delegation. "
                "Use --refine on run_keypoints_batch instead.",
                file=sys.stderr,
            )
            return 1
        return _delegate_refine_only(args)

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

    config = _load_config(args.config)
    method = _resolve_method(config, args.method)
    if args.model_source == "registry" and method not in {"yolo", "yolo_pose"}:
        print(
            f"Error: --model-source registry requires YOLO keypoints method, got '{method}'.",
            file=sys.stderr,
        )
        return 1
    explicit_background_pref = bool(args.require_background) or bool(args.no_require_background)
    if explicit_background_pref:
        require_background = bool(args.require_background) and not bool(args.no_require_background)
    else:
        require_background = method not in {"yolo", "yolo_pose"}
    require_crop = bool(args.require_crop) and not bool(args.no_require_crop)

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
            require_background=require_background,
            require_crop=require_crop,
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
            source=args.source,
            sql_prefilter=bool(args.source == "registry" and skip_existing),
            model_source=args.model_source,
            model_set_id=args.model_set_id,
            auto_approve_min_usable_rate=auto_approve_min_usable_rate,
            auto_approve_intended_use=args.auto_approve_intended_use,
            auto_approve_reviewer=args.auto_approve_reviewer,
        )

    plans: List[KeypointPlan] = []
    resolved_registry_path: Optional[Path] = None
    if args.source == "registry" or args.model_source == "registry":
        resolved_registry_path = _resolve_registry_path(args.registry)
    if args.source == "registry":
        if resolved_registry_path is None or not resolved_registry_path.exists():
            print(
                f"Error: registry database not found: {resolved_registry_path}",
                file=sys.stderr,
            )
            return 1
        registry_entries = _discover_registry_entries(
            registry_path=resolved_registry_path,
            scope_paths=roots + explicit_zarrs,
            rig_id=args.rig_id,
            arena_id=args.arena_id,
            camera_id=getattr(args, "camera_id_filter", None),
            path_contains=args.path_contains,
            skip_existing=skip_existing,
        )
        registry_zarrs = [entry.zarr_path for entry in registry_entries]
        camera_ids_by_zarr = {str(entry.zarr_path.resolve()): entry.camera_id for entry in registry_entries}
        if args.emit_paths:
            for p in registry_zarrs:
                print(p)
            return 0
        registry_plans = _build_plans_from_zarr(
            registry_zarrs,
            skip_existing=skip_existing,
            require_crop=require_crop,
            require_background=require_background,
            require_tuning=bool(args.require_tuning),
            refine_only=bool(args.refine_only),
        )
        for plan in registry_plans:
            camera_id = camera_ids_by_zarr.get(str(plan.zarr_path.resolve()))
            if camera_id is not None:
                plan.camera_id = camera_id
        plans.extend(registry_plans)
    else:
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

    resolved_models_by_plan: Dict[str, ResolvedModel] = {}
    if args.model_source == "registry" and runnable_plans:
        if resolved_registry_path is None or not resolved_registry_path.exists():
            print(
                f"Error: registry database not found for model resolution: {resolved_registry_path}",
                file=sys.stderr,
            )
            return 1
        if not args.json and not args.quiet:
            print(f"Pre-resolving registry models for {len(runnable_plans)} recording plan(s)...")

        def _emit_model_resolution_event(event_payload: Dict[str, Any]) -> None:
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
            event = event_name
            if event == "model_resolution_start":
                print(f"{prefix}Resolving model for {event_payload.get('recording')}")
                return
            if event == "model_resolution_cached":
                print(
                    f"{prefix}Using cached model for {event_payload.get('recording')}: "
                    f"{event_payload.get('selected_model_path')}"
                )
                return
            if event == "model_resolution_ok":
                print(
                    f"{prefix}Resolved model for {event_payload.get('recording')}: "
                    f"{event_payload.get('selected_model_path')}"
                )
                return
            if event == "model_resolution_failed":
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
            runnable_after_resolution: List[KeypointPlan] = []
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
                                "status": "failed",
                                "zarr": str(plan.zarr_path),
                                "error": f"model resolution failed: {error}",
                            }
                        )
                    )
                else:
                    print(f"Keypoints failed for {plan.zarr_path}: model resolution failed: {error}")
                if logger is not None:
                    logger.log(
                        "keypoints_failed",
                        zarr=str(plan.zarr_path),
                        error=f"model resolution failed: {error}",
                    )
            runnable_plans = runnable_after_resolution

    console = Console() if (Console is not None and not args.json and not args.quiet) else None
    progress = _progress(console if not args.json else None, total=len(runnable_plans))
    quiet = bool(args.quiet or args.json)
    task_label = "Refining keypoints" if args.refine_only else "Running keypoints"

    if progress is None:
        for plan in runnable_plans:
            try:
                result = _run_plan(
                    plan,
                    config=config,
                    method=method,
                    scheduler=args.scheduler,
                    num_workers=args.num_workers,
                    quiet=quiet,
                    dask_progress=bool(args.dask_progress and not args.quiet and not args.json),
                    refine=bool(args.refine),
                    refine_only=bool(args.refine_only),
                    json_output=bool(args.json),
                    resolved_model=resolved_models_by_plan.get(str(plan.zarr_path.resolve())),
                    auto_approve_min_usable_rate=auto_approve_min_usable_rate,
                    auto_approve_intended_use=str(args.auto_approve_intended_use),
                    auto_approve_reviewer=args.auto_approve_reviewer,
                    registry_path_for_sync=(resolved_registry_path or args.registry),
                )
                ok += 1
                if args.json:
                    print(json.dumps(result, sort_keys=True))
                if logger is not None:
                    logger.log(
                        "keypoints_ok",
                        zarr=str(plan.zarr_path),
                        results=result,
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
                    result = _run_plan(
                        plan,
                        config=config,
                        method=method,
                        scheduler=args.scheduler,
                        num_workers=args.num_workers,
                        quiet=quiet,
                        dask_progress=bool(args.dask_progress and not args.quiet and not args.json),
                        refine=bool(args.refine),
                        refine_only=bool(args.refine_only),
                        json_output=bool(args.json),
                        resolved_model=resolved_models_by_plan.get(str(plan.zarr_path.resolve())),
                        auto_approve_min_usable_rate=auto_approve_min_usable_rate,
                        auto_approve_intended_use=str(args.auto_approve_intended_use),
                        auto_approve_reviewer=args.auto_approve_reviewer,
                        registry_path_for_sync=(resolved_registry_path or args.registry),
                    )
                    ok += 1
                    if args.json:
                        print(json.dumps(result, sort_keys=True))
                    if logger is not None:
                        logger.log(
                            "keypoints_ok",
                            zarr=str(plan.zarr_path),
                            results=result,
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
