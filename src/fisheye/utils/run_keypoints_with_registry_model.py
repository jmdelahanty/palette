#!/usr/bin/env python3
"""Resolve a pose model from registry, run keypoint inference, and persist resolution provenance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import zarr

from fisheye.detection.detect_keypoints_yolo import DEFAULT_POSE_SCHEMA_NAME, detect_keypoints_yolo
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.flat_roi_cache import open_flat_roi_cache
from fisheye.utils.model_resolution_provenance import build_model_resolution_payload
from fisheye.registry.model_resolution import Candidate, TargetProfile, load_candidates, load_target_profile, resolve_recording_id

ROI_CACHE_STAGING_RECOMMENDED_MIN_BYTES = 5 * 1024**3
ROI_CACHE_STAGING_BENCHMARK_NOTE = (
    "GoodCopBadCop L4 benchmark: staging a 33.4 GiB flat cache to node scratch "
    "improved keypoint inference from ~212 to ~276 poses/s and remained faster "
    "end-to-end after a ~46s copy."
)


def _resolve_output(recording_dir: Path, explicit_output: Optional[Path]) -> Path:
    if explicit_output is not None:
        return explicit_output.expanduser().resolve()
    return (recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr").resolve()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_roi_cache_source_tier(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    resolved = str(path.expanduser().resolve())
    if resolved.startswith("/scratch/") or resolved.startswith("/tmp/"):
        return "node_scratch"
    if resolved.startswith("/groups/") or resolved.startswith("/misc/public/"):
        return "prfs_workflow_scratch"
    return "unknown"


def _default_roi_cache_staging_dir() -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID")
    if job_id and Path(f"/scratch/{user}").is_dir():
        return Path("/scratch") / user / job_id / "palette_roi_cache_stage"
    return Path(os.environ.get("TMPDIR") or "/tmp") / f"palette_roi_cache_stage_{os.getpid()}"


def _resolve_manifest_payload_path(manifest_path: Path, payload_value: object) -> Path:
    payload_text = str(payload_value or "")
    if not payload_text:
        raise ValueError(f"Flat ROI cache manifest is missing array.bin_path: {manifest_path}")
    payload_path = Path(payload_text).expanduser()
    if payload_path.is_absolute():
        return payload_path
    return manifest_path.parent / payload_path


def _flat_roi_cache_payload_info(manifest_path: Path) -> dict[str, Any]:
    """Return best-effort flat-cache payload metadata for policy provenance."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        array = payload.get("array")
        if not isinstance(array, dict):
            return {"payload_inspection_status": "missing_array_metadata"}
        source_bin = _resolve_manifest_payload_path(manifest_path, array.get("bin_path")).resolve()
        if not source_bin.exists():
            return {
                "payload_inspection_status": "payload_missing",
                "source_bin_path": str(source_bin),
            }
        return {
            "payload_inspection_status": "ok",
            "source_bin_path": str(source_bin),
            "payload_size_bytes": int(source_bin.stat().st_size),
        }
    except FileNotFoundError:
        return {"payload_inspection_status": "manifest_missing"}
    except Exception as exc:
        return {
            "payload_inspection_status": "error",
            "payload_inspection_error": str(exc),
        }


def _staging_recommendation_payload(
    *,
    manifest_path: Path,
    source_tier: Optional[str],
    payload_size_bytes: Optional[int],
) -> dict[str, Any]:
    is_shared_cache = source_tier == "prfs_workflow_scratch"
    recommended = bool(
        is_shared_cache
        and payload_size_bytes is not None
        and int(payload_size_bytes) >= ROI_CACHE_STAGING_RECOMMENDED_MIN_BYTES
    )
    return {
        "staging_recommended": recommended,
        "staging_recommendation_reason": (
            "large_prfs_flat_cache"
            if recommended
            else "not_required_by_size_or_source_tier"
        ),
        "staging_recommendation_min_bytes": ROI_CACHE_STAGING_RECOMMENDED_MIN_BYTES,
        "staging_recommendation_basis": ROI_CACHE_STAGING_BENCHMARK_NOTE,
        "staging_recommendation_source_tier": source_tier,
        "staging_recommendation_manifest_path": str(manifest_path),
    }


def _copy_file_with_timing(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    started_utc = _now_utc()
    started = time.perf_counter()
    shutil.copy2(source, destination)
    seconds = float(time.perf_counter() - started)
    size_bytes = int(destination.stat().st_size)
    return {
        "source_path": str(source),
        "destination_path": str(destination),
        "started_at_utc": started_utc,
        "finished_at_utc": _now_utc(),
        "duration_seconds": seconds,
        "size_bytes": size_bytes,
        "mib_per_second": (size_bytes / (1024 * 1024)) / seconds if seconds > 0 else None,
    }


def _stage_flat_roi_cache_manifest(
    manifest_path: Path,
    *,
    staging_dir: Optional[Path] = None,
) -> tuple[Path, dict[str, Any]]:
    """Copy a flat ROI cache manifest and payload to node-local scratch."""

    source_manifest = manifest_path.expanduser().resolve()
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    array = payload.get("array")
    if not isinstance(array, dict):
        raise ValueError(f"Flat ROI cache manifest is missing array metadata: {source_manifest}")
    source_bin = _resolve_manifest_payload_path(source_manifest, array.get("bin_path")).resolve()
    if not source_bin.exists():
        raise FileNotFoundError(f"Flat ROI cache payload not found: {source_bin}")
    source_tier = _infer_roi_cache_source_tier(source_manifest)
    payload_size_bytes = int(source_bin.stat().st_size)

    target_dir = (staging_dir.expanduser() if staging_dir is not None else _default_roi_cache_staging_dir()).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = target_dir / source_manifest.name
    local_bin = target_dir / source_bin.name
    if local_manifest == source_manifest or local_bin == source_bin:
        raise ValueError(
            "ROI cache staging directory resolves to the source cache directory; "
            "choose a distinct node-local scratch directory."
        )

    tmp_bin = local_bin.with_name(f"{local_bin.name}.tmp.{os.getpid()}")
    tmp_manifest = local_manifest.with_name(f"{local_manifest.name}.tmp.{os.getpid()}")

    bin_copy = _copy_file_with_timing(source_bin, tmp_bin)
    os.replace(tmp_bin, local_bin)

    staged_payload = dict(payload)
    staged_array = dict(array)
    staged_payload["manifest_path"] = str(local_manifest)
    staged_array["bin_path"] = local_bin.name
    staged_payload["array"] = staged_array
    staging_details: dict[str, Any] = {
        "schema": "palette_roi_cache_staging_v1",
        "policy": "node_scratch_staged_flat_cache",
        "staged": True,
        "stage_to_scratch_requested": True,
        "requested_manifest_path": str(source_manifest),
        "effective_manifest_path": str(local_manifest),
        "source_bin_path": str(source_bin),
        "effective_bin_path": str(local_bin),
        "source_tier": source_tier,
        "effective_source_tier": "node_scratch",
        "payload_size_bytes": payload_size_bytes,
        "staging_dir": str(target_dir),
        "host": socket.gethostname(),
        "lsb_jobid": os.environ.get("LSB_JOBID"),
        "payload_copy": {
            **bin_copy,
            "destination_path": str(local_bin),
        },
        "manifest_publish_policy": "payload_first_manifest_last",
        **_staging_recommendation_payload(
            manifest_path=source_manifest,
            source_tier=source_tier,
            payload_size_bytes=payload_size_bytes,
        ),
    }
    staged_payload["staging"] = staging_details
    tmp_manifest.write_text(json.dumps(staged_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, local_manifest)

    cache = open_flat_roi_cache(local_manifest)
    try:
        staging_details["validation_status"] = "ok"
        staging_details["validated_shape"] = list(cache.shape)
        staging_details["validated_dtype"] = str(cache.dtype)
    finally:
        cache.close()
    return local_manifest, staging_details


def _prepare_roi_cache_manifest(
    manifest_path: Optional[Path],
    *,
    stage_to_scratch: bool,
    staging_dir: Optional[Path],
) -> tuple[Optional[Path], dict[str, Any]]:
    if manifest_path is None:
        if stage_to_scratch:
            raise ValueError("--stage-roi-cache-to-scratch requires --roi-cache-manifest.")
        return None, {}
    requested = manifest_path.expanduser().resolve()
    source_tier = _infer_roi_cache_source_tier(requested)
    if not stage_to_scratch:
        payload_info = _flat_roi_cache_payload_info(requested)
        payload_size_bytes = payload_info.get("payload_size_bytes")
        if payload_size_bytes is not None:
            payload_size_bytes = int(payload_size_bytes)
        return requested, {
            "schema": "palette_roi_cache_staging_v1",
            "policy": "direct_manifest_read",
            "staged": False,
            "stage_to_scratch_requested": False,
            "requested_manifest_path": str(requested),
            "effective_manifest_path": str(requested),
            "source_tier": source_tier,
            "effective_source_tier": source_tier,
            "validation_status": "not_revalidated_by_wrapper",
            **payload_info,
            **_staging_recommendation_payload(
                manifest_path=requested,
                source_tier=source_tier,
                payload_size_bytes=payload_size_bytes,
            ),
        }
    return _stage_flat_roi_cache_manifest(requested, staging_dir=staging_dir)


def _roi_cache_source_crop_run_name(manifest_path: Optional[Path]) -> Optional[str]:
    if manifest_path is None:
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    source = payload.get("source")
    if not isinstance(source, dict):
        return None
    text = str(source.get("crop_run_name") or "").strip()
    return text or None


def pick_best_keypoint_candidate(candidates: list[Candidate], *, require_unique: bool) -> Candidate:
    if not candidates:
        raise SystemExit("No pose model candidates found.")
    best = candidates[0]
    if require_unique and len(candidates) > 1:
        if abs(candidates[0].weighted_score - candidates[1].weighted_score) < 1e-12:
            raise SystemExit("Top candidate score tied; rerun with --set-id to choose deterministically.")
    return best


def _pick_best_candidate(candidates: list[Candidate], *, require_unique: bool) -> Candidate:
    return pick_best_keypoint_candidate(candidates, require_unique=require_unique)


def build_keypoint_resolution_payload(
    *,
    args: argparse.Namespace,
    argv: Optional[list[str]],
    recording_dir: Path,
    output_path: Path,
    registry_path: Path,
    recording_id: str,
    target: TargetProfile,
    selected: Candidate,
    candidates: list[Candidate],
    top_k: int,
) -> dict[str, Any]:
    target_payload = asdict(target)

    def _candidate_payload(item: Candidate) -> dict[str, Any]:
        return {
            "run_id": item.run_id,
            "set_id": item.set_id,
            "model_path": item.model_path,
            "model_sha256": item.model_sha256,
            "score": item.weighted_score,
            "created_utc": item.created_utc,
            "status": item.status,
            "dataset_count": item.dataset_count,
            "feature_match_counts": item.feature_match_counts,
            "feature_weights_used": item.feature_weights_used,
        }

    selected_payload = _candidate_payload(selected)
    candidate_payloads = [_candidate_payload(item) for item in candidates[: max(0, int(top_k))]]

    return build_model_resolution_payload(
        tool="fisheye.utils.run_keypoints_with_registry_model",
        args=args,
        argv=argv,
        task="pose",
        registry_path=registry_path,
        recording_id=recording_id,
        target=target_payload,
        selected=selected_payload,
        candidates=candidate_payloads,
        parameters={
            "set_id_filter": args.set_id,
            "require_unique": bool(args.require_unique),
            "top_k": int(args.top_k),
            "include_non_success": bool(args.include_non_success),
            "dry_run": bool(args.dry_run),
            "cpu": bool(args.cpu),
            "run_name": args.run_name,
            "crop_run": args.crop_run,
            "pose_schema": args.pose_schema,
            "batch_size": args.batch_size,
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "mask_threshold": args.mask_threshold,
            "roi_cache_policy": args.roi_cache_policy,
            "roi_cache_dir": str(args.roi_cache_dir) if args.roi_cache_dir else None,
            "roi_cache_manifest": str(args.roi_cache_manifest) if args.roi_cache_manifest else None,
            "stage_roi_cache_to_scratch": bool(args.stage_roi_cache_to_scratch),
            "roi_cache_staging_dir": str(args.roi_cache_staging_dir) if args.roi_cache_staging_dir else None,
            "profile_timings": bool(args.profile_timings),
        },
        inputs={
            "recording_dir": str(recording_dir),
            "output_zarr": str(output_path),
            "recording_id": recording_id,
            "target": target_payload,
        },
        artifacts={
            "selected_model": selected_payload,
            "candidate_models": candidate_payloads,
            "output_zarr": str(output_path),
        },
    )


def _resolution_payload(
    **kwargs: Any,
) -> dict[str, Any]:
    return build_keypoint_resolution_payload(**kwargs)


def write_keypoint_model_resolution_provenance(
    *,
    zarr_path: Path,
    run_name: str,
    payload: dict[str, Any],
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r+")
    keypoint_parent = root.get("keypoints_runs")
    if keypoint_parent is None or run_name not in keypoint_parent:
        raise RuntimeError(f"keypoint run not found for provenance annotation: keypoints_runs/{run_name}")

    keypoint_group = keypoint_parent[run_name]
    selected = payload.get("selected", {}) if isinstance(payload.get("selected"), dict) else {}
    attrs = dict(keypoint_group.attrs)
    attrs["model_resolution_mode"] = "registry"
    attrs["model_resolution_task"] = "pose"
    attrs["model_resolution_registry_path"] = payload.get("registry_path")
    attrs["model_resolution_recording_id"] = payload.get("recording_id")
    attrs["model_resolution_selected_run_id"] = selected.get("run_id")
    attrs["model_resolution_selected_set_id"] = selected.get("set_id")
    attrs["model_resolution_selected_model_path"] = selected.get("model_path")
    attrs["model_resolution_selected_score"] = selected.get("score")
    attrs["model_resolution_selected_created_utc"] = selected.get("created_utc")
    attrs["model_resolution_resolved_at_utc"] = payload.get("resolved_at_utc")
    attrs["model_resolution_candidates_json"] = json.dumps(payload.get("candidates", []), sort_keys=True)

    provenance = attrs.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["model_resolution"] = payload
    attrs["provenance"] = provenance
    keypoint_group.attrs.put(attrs)


def _write_model_resolution_provenance(
    *,
    zarr_path: Path,
    run_name: str,
    payload: dict[str, Any],
) -> None:
    write_keypoint_model_resolution_provenance(zarr_path=zarr_path, run_name=run_name, payload=payload)


@dataclass(frozen=True)
class KeypointRegistryResult:
    ok: bool
    status: str
    recording_dir: str
    output_zarr: str
    registry_path: str
    reason: Optional[str] = None
    error: Optional[str] = None
    remediation: Optional[str] = None
    selected_model_path: Optional[str] = None
    selected_run_id: Optional[str] = None
    selected_set_id: Optional[str] = None
    keypoint_run: Optional[str] = None
    resolved_at_utc: Optional[str] = None
    resolution_payload: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "recording_dir": self.recording_dir,
            "output_zarr": self.output_zarr,
            "registry_path": self.registry_path,
            "reason": self.reason,
            "error": self.error,
            "remediation": self.remediation,
            "selected_model_path": self.selected_model_path,
            "selected_run_id": self.selected_run_id,
            "selected_set_id": self.selected_set_id,
            "keypoint_run": self.keypoint_run,
            "resolved_at_utc": self.resolved_at_utc,
        }
        if self.resolution_payload is not None:
            payload["resolution_payload"] = self.resolution_payload
        return payload


def _failure_result(
    *,
    reason: str,
    error: str,
    remediation: str,
    recording_dir: Path,
    output_path: Path,
    registry_path: Path,
    selected_model_path: Optional[str] = None,
    selected_run_id: Optional[str] = None,
    selected_set_id: Optional[str] = None,
    keypoint_run: Optional[str] = None,
    resolved_at_utc: Optional[str] = None,
    resolution_payload: Optional[dict[str, Any]] = None,
) -> KeypointRegistryResult:
    return KeypointRegistryResult(
        ok=False,
        status="failed",
        reason=reason,
        error=error,
        remediation=remediation,
        recording_dir=str(recording_dir),
        output_zarr=str(output_path),
        registry_path=str(registry_path),
        selected_model_path=selected_model_path,
        selected_run_id=selected_run_id,
        selected_set_id=selected_set_id,
        keypoint_run=keypoint_run,
        resolved_at_utc=resolved_at_utc,
        resolution_payload=resolution_payload,
    )


def run_keypoints_with_registry_model(
    *,
    recording_dir: Path,
    output: Optional[Path] = None,
    registry: Optional[Path] = None,
    set_id: Optional[str] = None,
    require_unique: bool = False,
    top_k: int = 5,
    include_non_success: bool = False,
    dry_run: bool = False,
    run_name: Optional[str] = None,
    crop_run: Optional[str] = None,
    pose_schema: str = DEFAULT_POSE_SCHEMA_NAME,
    batch_size: int = 256,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 1,
    mask_threshold: float = 0.5,
    roi_cache_policy: str = "auto",
    roi_cache_dir: Optional[Path] = None,
    roi_cache_manifest: Optional[Path] = None,
    stage_roi_cache_to_scratch: bool = False,
    roi_cache_staging_dir: Optional[Path] = None,
    profile_timings: bool = False,
    progress_jsonl: Optional[Path] = None,
    progress_every_batches: int = 1,
    input_mode: str = "numpy-list",
    cpu: bool = False,
    verbose: bool = False,
    argv: Optional[list[str]] = None,
    cli_provenance: Optional[Mapping[str, Any]] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
) -> KeypointRegistryResult:
    resolved_recording_dir = recording_dir.expanduser().resolve()
    registry_path = (registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    output_path = _resolve_output(resolved_recording_dir, output)

    payload_args = argparse.Namespace(
        recording_dir=resolved_recording_dir,
        output=output,
        registry=registry_path,
        set_id=set_id,
        require_unique=bool(require_unique),
        top_k=int(top_k),
        include_non_success=bool(include_non_success),
        dry_run=bool(dry_run),
        run_name=run_name,
        crop_run=crop_run,
        pose_schema=pose_schema,
        batch_size=int(batch_size),
        device=device,
        imgsz=imgsz,
        conf=float(conf),
        iou=float(iou),
        max_det=int(max_det),
        mask_threshold=float(mask_threshold),
        roi_cache_policy=roi_cache_policy,
        roi_cache_dir=roi_cache_dir,
        roi_cache_manifest=roi_cache_manifest,
        stage_roi_cache_to_scratch=bool(stage_roi_cache_to_scratch),
        roi_cache_staging_dir=roi_cache_staging_dir,
        profile_timings=bool(profile_timings),
        progress_jsonl=progress_jsonl,
        progress_every_batches=int(progress_every_batches),
        input_mode=input_mode,
        cpu=bool(cpu),
        verbose=bool(verbose),
    )

    try:
        registry_db = Registry(registry_path)
    except Exception as exc:
        return _failure_result(
            reason="registry_open_failed",
            error=str(exc),
            remediation="Verify --registry points to a readable palette registry SQLite file.",
            recording_dir=resolved_recording_dir,
            output_path=output_path,
            registry_path=registry_path,
        )

    try:
        recording_id = resolve_recording_id(
            registry_db,
            recording_id=None,
            recording_dir=resolved_recording_dir,
        )
        target = load_target_profile(registry_db, recording_id)
        candidates = load_candidates(
            registry_db,
            target=target,
            task="pose",
            set_id_filter=set_id,
            include_non_success=bool(include_non_success),
        )
    except Exception as exc:
        return _failure_result(
            reason="model_resolution_failed",
            error=str(exc),
            remediation="Verify registry metadata for this recording and rerun with --include-non-success or --set-id as needed.",
            recording_dir=resolved_recording_dir,
            output_path=output_path,
            registry_path=registry_path,
        )
    finally:
        registry_db.close()

    try:
        best = pick_best_keypoint_candidate(candidates, require_unique=bool(require_unique))
    except SystemExit as exc:
        return _failure_result(
            reason="candidate_selection_failed",
            error=str(exc),
            remediation="Pass --set-id to pin a model set or remove --require-unique.",
            recording_dir=resolved_recording_dir,
            output_path=output_path,
            registry_path=registry_path,
        )

    payload = build_keypoint_resolution_payload(
        args=payload_args,
        argv=argv,
        recording_dir=resolved_recording_dir,
        output_path=output_path,
        registry_path=registry_path,
        recording_id=recording_id,
        target=target,
        selected=best,
        candidates=candidates,
        top_k=int(top_k),
    )
    selected_payload = payload.get("selected") if isinstance(payload.get("selected"), dict) else {}
    selected_model_path = selected_payload.get("model_path") if isinstance(selected_payload.get("model_path"), str) else None
    selected_model_sha256 = selected_payload.get("model_sha256") if isinstance(selected_payload.get("model_sha256"), str) else None
    selected_run_id = selected_payload.get("run_id") if isinstance(selected_payload.get("run_id"), str) else None
    selected_set_id = selected_payload.get("set_id") if isinstance(selected_payload.get("set_id"), str) else None
    resolved_at_utc = payload.get("resolved_at_utc") if isinstance(payload.get("resolved_at_utc"), str) else None

    if dry_run:
        return KeypointRegistryResult(
            ok=True,
            status="dry_run",
            recording_dir=str(resolved_recording_dir),
            output_zarr=str(output_path),
            registry_path=str(registry_path),
            selected_model_path=selected_model_path,
            selected_run_id=selected_run_id,
            selected_set_id=selected_set_id,
            resolved_at_utc=resolved_at_utc,
            resolution_payload=payload,
        )

    try:
        resolved_device = "cpu" if cpu else device
        effective_roi_cache_manifest, roi_cache_staging_details = _prepare_roi_cache_manifest(
            roi_cache_manifest,
            stage_to_scratch=bool(stage_roi_cache_to_scratch),
            staging_dir=roi_cache_staging_dir,
        )
        effective_crop_run = crop_run or _roi_cache_source_crop_run_name(effective_roi_cache_manifest)
        effective_run_provenance = run_provenance if run_provenance is not None else cli_provenance
        if effective_run_provenance is None:
            effective_run_provenance = build_run_provenance(
                command="fisheye.utils.run_keypoints_with_registry_model",
                params={
                    **vars(payload_args),
                    "recording_dir": resolved_recording_dir,
                    "output": output_path,
                    "registry": registry_path,
                    "selected_model_path": selected_model_path,
                    "selected_run_id": selected_run_id,
                    "selected_set_id": selected_set_id,
                    "effective_roi_cache_manifest": effective_roi_cache_manifest,
                },
                input_run_ids={
                    "crop_run": effective_crop_run,
                    "model_run": selected_run_id,
                    "model_set": selected_set_id,
                },
                cwd=Path.cwd(),
            )
        keypoint_run = detect_keypoints_yolo(
            zarr_path=str(output_path),
            model_path=best.model_path,
            model_sha256=selected_model_sha256,
            run_name=run_name,
            crop_run=crop_run,
            pose_schema=pose_schema,
            batch_size=batch_size,
            device=resolved_device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=bool(verbose),
            mask_threshold=mask_threshold,
            roi_cache_policy=roi_cache_policy,
            roi_cache_dir=roi_cache_dir,
            roi_cache_manifest=effective_roi_cache_manifest,
            roi_cache_source_tier=roi_cache_staging_details.get("effective_source_tier"),
            roi_cache_staged_to_node_scratch=bool(roi_cache_staging_details.get("staged", False)),
            roi_cache_staging_details=roi_cache_staging_details or None,
            input_mode=input_mode,
            profile_timings=bool(profile_timings),
            progress_jsonl=progress_jsonl,
            progress_every_batches=progress_every_batches,
            registry=registry_path,
            cli_provenance=effective_run_provenance,
            run_provenance=effective_run_provenance,
        )
        if not keypoint_run:
            raise RuntimeError("Keypoint inference did not create a run; model resolution provenance cannot be written.")
        write_keypoint_model_resolution_provenance(
            zarr_path=output_path,
            run_name=keypoint_run,
            payload=payload,
        )
    except Exception as exc:
        return _failure_result(
            reason="keypoint_inference_failed",
            error=str(exc),
            remediation="Inspect model/config inputs and rerun with --dry-run --json to verify resolved model selection.",
            recording_dir=resolved_recording_dir,
            output_path=output_path,
            registry_path=registry_path,
            selected_model_path=selected_model_path,
            selected_run_id=selected_run_id,
            selected_set_id=selected_set_id,
            resolved_at_utc=resolved_at_utc,
            resolution_payload=payload,
        )

    return KeypointRegistryResult(
        ok=True,
        status="ok",
        recording_dir=str(resolved_recording_dir),
        output_zarr=str(output_path),
        registry_path=str(registry_path),
        selected_model_path=selected_model_path,
        selected_run_id=selected_run_id,
        selected_set_id=selected_set_id,
        keypoint_run=keypoint_run,
        resolved_at_utc=resolved_at_utc,
        resolution_payload=payload,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True, help="Recording directory to process.")
    parser.add_argument("--output", type=Path, help="Optional explicit output zarr path.")
    parser.add_argument("--registry", type=Path, help="Optional registry sqlite path.")
    parser.add_argument("--set-id", type=str, help="Optional set filter during model resolution.")
    parser.add_argument("--require-unique", action="store_true", help="Fail if top scores tie.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to store in provenance.")
    parser.add_argument("--include-non-success", action="store_true", help="Include non-success training runs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve model only; do not run keypoints.")

    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit keypoints run name.")
    parser.add_argument("--crop-run", type=str, default=None, help="Optional explicit crop run name.")
    parser.add_argument(
        "--pose-schema",
        type=str,
        default=DEFAULT_POSE_SCHEMA_NAME,
        help=(
            "Pose schema from configs/fisheye/pose_schemas to stamp on the output run. "
            f"Defaults to detect_keypoints_yolo's default ({DEFAULT_POSE_SCHEMA_NAME}) when omitted."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Optional keypoint batch size override.")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional pose inference image size override.")
    parser.add_argument("--conf", type=float, default=0.25, help="Optional confidence threshold override.")
    parser.add_argument("--iou", type=float, default=0.5, help="Optional IoU threshold override.")
    parser.add_argument("--max-det", type=int, default=1, help="Optional max detections override.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Optional compatibility threshold.")
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-cache-dir",
        type=Path,
        default=None,
        help="Optional scratch directory for temporary ROI caches.",
    )
    parser.add_argument(
        "--roi-cache-manifest",
        type=Path,
        default=None,
        help="Optional flat_bin_v1 ROI cache manifest to consume directly.",
    )
    parser.add_argument(
        "--stage-roi-cache-to-scratch",
        action="store_true",
        help="Copy --roi-cache-manifest and payload to node-local scratch before inference.",
    )
    parser.add_argument(
        "--roi-cache-staging-dir",
        type=Path,
        default=None,
        help="Optional directory for staged flat ROI cache files; defaults to /scratch/$USER/$LSB_JOBID when available.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Collect per-stage timing diagnostics and store them in the output keypoint run attrs.",
    )
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file for live keypoint progress events.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=1,
        help="Write one progress JSONL event every N completed batches (default: 1).",
    )
    parser.add_argument(
        "--input-mode",
        choices=("numpy-list", "tensor", "auto"),
        default="numpy-list",
        help="Ultralytics input preparation mode (default: legacy numpy-list).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output.")
    parser.add_argument("--json", action="store_true", help="Print resolved payload JSON.")
    args = parser.parse_args(argv)

    result = run_keypoints_with_registry_model(
        recording_dir=args.recording_dir,
        output=args.output,
        registry=args.registry,
        set_id=args.set_id,
        require_unique=bool(args.require_unique),
        top_k=int(args.top_k),
        include_non_success=bool(args.include_non_success),
        dry_run=bool(args.dry_run),
        run_name=args.run_name,
        crop_run=args.crop_run,
        pose_schema=args.pose_schema,
        batch_size=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        mask_threshold=args.mask_threshold,
        roi_cache_policy=args.roi_cache_policy,
        roi_cache_dir=args.roi_cache_dir,
        roi_cache_manifest=args.roi_cache_manifest,
        stage_roi_cache_to_scratch=bool(args.stage_roi_cache_to_scratch),
        roi_cache_staging_dir=args.roi_cache_staging_dir,
        profile_timings=bool(args.profile_timings),
        progress_jsonl=args.progress_jsonl,
        progress_every_batches=args.progress_every_batches,
        input_mode=args.input_mode,
        cpu=bool(args.cpu),
        verbose=bool(args.verbose),
        argv=argv,
    )

    if args.json or args.dry_run:
        if result.resolution_payload is not None:
            print(json.dumps(result.resolution_payload, indent=2, sort_keys=True))
        else:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if result.status == "dry_run":
        return 0

    if not result.ok:
        print("Keypoint run failed")
        print(f"  recording_dir: {result.recording_dir}")
        print(f"  output_zarr: {result.output_zarr}")
        print(f"  reason: {result.reason or 'unknown'}")
        if result.error:
            print(f"  error: {result.error}")
        if result.remediation:
            print(f"  remediation: {result.remediation}")
        return 1

    print("Model resolution provenance written")
    print(f"  output_zarr: {result.output_zarr}")
    print(f"  keypoint_run: {result.keypoint_run}")
    print(f"  selected_model: {result.selected_model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
