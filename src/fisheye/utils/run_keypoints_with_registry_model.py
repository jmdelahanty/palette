#!/usr/bin/env python3
"""Resolve a pose model from registry, run keypoint inference, and persist resolution provenance."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import zarr

from fisheye.detection.detect_keypoints_yolo import (
    DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    DEFAULT_KEYPOINT_OUTPUT_PARENT,
    DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    KEYPOINT_OUTPUT_PARENTS,
    detect_keypoints_yolo,
)
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.pose.schema import schema_from_package, undirected_edge_topology
from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.flat_roi_cache import (
    open_flat_roi_cache,
    stage_flat_roi_cache_manifest,
)
from fisheye.shared.model_input_transform import MODEL_INPUT_TRANSFORM_CHOICES
from fisheye.shared.pose_model_schema_binding import (
    resolve_registered_pose_model_schema_binding,
)
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
    job_index = os.environ.get("LSB_JOBINDEX")
    user_scratch = Path(f"/scratch/{user}")
    if job_id and user_scratch.is_dir() and os.access(user_scratch, os.W_OK | os.X_OK):
        work_unit = f"{job_id}_{job_index}" if job_index else job_id
        return user_scratch / work_unit / "palette_roi_cache_stage"
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


def _stage_flat_roi_cache_manifest(
    manifest_path: Path,
    *,
    staging_dir: Optional[Path] = None,
) -> tuple[Path, dict[str, Any]]:
    """Copy and authenticate a flat ROI cache to node-local scratch once."""

    source_manifest = manifest_path.expanduser().resolve()
    source_tier = _infer_roi_cache_source_tier(source_manifest)
    target_dir = (
        staging_dir.expanduser()
        if staging_dir is not None
        else _default_roi_cache_staging_dir()
    ).resolve()
    local_manifest, shared_details = stage_flat_roi_cache_manifest(
        source_manifest,
        staging_dir=target_dir,
    )
    staging_details = dict(shared_details)
    payload_size_bytes = int(staging_details["copy"]["size_bytes"])
    staging_details.update(
        {
            "stage_to_scratch_requested": True,
            "source_tier": source_tier,
            "effective_source_tier": "node_scratch",
            "payload_size_bytes": payload_size_bytes,
            **_staging_recommendation_payload(
                manifest_path=source_manifest,
                source_tier=source_tier,
                payload_size_bytes=payload_size_bytes,
            ),
        }
    )
    # Preserve the historical field name for downstream timing reports while
    # the shared helper owns the exact copy/hash semantics.
    staging_details["payload_copy"] = dict(staging_details["copy"])
    staged_payload = json.loads(local_manifest.read_text(encoding="utf-8"))
    staged_payload["staging"] = staging_details
    tmp_manifest = local_manifest.with_name(
        f"{local_manifest.name}.tmp.{os.getpid()}"
    )
    tmp_manifest.write_text(
        json.dumps(staged_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def validate_pose_candidate_bindings(
    registry: Registry,
    candidates: list[Candidate],
    *,
    expected_pose_schema: Optional[str] = None,
) -> tuple[list[Candidate], dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Reject internally inconsistent pose candidates before score selection."""
    expected_labels: Optional[list[str]] = None
    expected_edges: Optional[list[list[int]]] = None
    if expected_pose_schema is not None:
        schema = schema_from_package(expected_pose_schema)
        expected_labels = list(schema.node_names)
        expected_edges = [list(edge) for edge in schema.edges]
    compatible: list[Candidate] = []
    bindings: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            binding = resolve_registered_pose_model_schema_binding(
                registry,
                run_id=candidate.run_id,
                expected_set_id=candidate.set_id,
                expected_model_path=candidate.model_path,
                expected_model_sha256=candidate.model_sha256,
            )
        except Exception as exc:
            rejected.append(
                {
                    "run_id": candidate.run_id,
                    "set_id": candidate.set_id,
                    "reason": "pose_schema_binding_invalid",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        bound_schema = binding.get("pose_schema")
        if expected_labels is not None and (
            not isinstance(bound_schema, Mapping)
            or list(bound_schema.get("keypoint_labels") or []) != expected_labels
            or undirected_edge_topology(bound_schema.get("edges"))
            != undirected_edge_topology(expected_edges)
        ):
            rejected.append(
                {
                    "run_id": candidate.run_id,
                    "set_id": candidate.set_id,
                    "reason": "requested_pose_schema_mismatch",
                    "error_type": "PoseModelSchemaMismatch",
                    "error": (
                        f"Model binding does not match requested pose schema "
                        f"{expected_pose_schema!r}."
                    ),
                }
            )
            continue
        compatible.append(candidate)
        bindings[candidate.run_id] = binding
    return compatible, bindings, rejected


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
            "manifest_path": item.manifest_path,
            "manifest_sha256": item.manifest_sha256,
            "skeleton_id": item.skeleton_id,
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
    output_parent: str = DEFAULT_KEYPOINT_OUTPUT_PARENT,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
    keypoint_parent = root.get(output_parent)
    if keypoint_parent is None or run_name not in keypoint_parent:
        raise RuntimeError(f"keypoint run not found for provenance annotation: {output_parent}/{run_name}")

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
    output_parent: str = DEFAULT_KEYPOINT_OUTPUT_PARENT,
) -> None:
    write_keypoint_model_resolution_provenance(
        zarr_path=zarr_path,
        run_name=run_name,
        payload=payload,
        output_parent=output_parent,
    )


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
    keypoint_parent: Optional[str] = None
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
            "keypoint_parent": self.keypoint_parent,
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
    keypoint_parent: Optional[str] = None,
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
        keypoint_parent=keypoint_parent,
        resolved_at_utc=resolved_at_utc,
        resolution_payload=resolution_payload,
    )


def run_keypoints_with_registry_model(
    *,
    recording_dir: Path,
    output: Optional[Path] = None,
    registry: Optional[Path] = None,
    set_id: Optional[str] = None,
    model_run_id: Optional[str] = None,
    require_unique: bool = False,
    top_k: int = 5,
    include_non_success: bool = False,
    dry_run: bool = False,
    run_name: Optional[str] = None,
    output_parent: str = DEFAULT_KEYPOINT_OUTPUT_PARENT,
    crop_run: Optional[str] = None,
    pose_schema: Optional[str] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    model_input_size: Optional[int] = None,
    expected_model_stride: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 1,
    mask_threshold: float = 0.5,
    roi_cache_policy: str = "auto",
    roi_cache_dir: Optional[Path] = None,
    roi_cache_manifest: Optional[Path] = None,
    roi_cache_expected_archive_path: Optional[Path] = None,
    source_crop_row_start: Optional[int] = None,
    source_crop_row_stop: Optional[int] = None,
    stage_roi_cache_to_scratch: bool = False,
    roi_cache_staging_dir: Optional[Path] = None,
    profile_timings: bool = False,
    progress_jsonl: Optional[Path] = None,
    progress_every_batches: int = 1,
    input_mode: str = "numpy-list",
    model_input_transform_mode: str = "auto",
    coordinate_contract_mode: str = "canonical",
    require_training_materialization_binding: bool = False,
    keypoint_roi_shard_rows: Optional[int] = DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    keypoint_frame_shard_rows: int = DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
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
        model_run_id=model_run_id,
        require_unique=bool(require_unique),
        top_k=int(top_k),
        include_non_success=bool(include_non_success),
        dry_run=bool(dry_run),
        run_name=run_name,
        output_parent=output_parent,
        crop_run=crop_run,
        pose_schema=pose_schema,
        batch_size=int(batch_size),
        device=device,
        imgsz=imgsz,
        model_input_size=model_input_size,
        expected_model_stride=expected_model_stride,
        conf=float(conf),
        iou=float(iou),
        max_det=int(max_det),
        mask_threshold=float(mask_threshold),
        roi_cache_policy=roi_cache_policy,
        roi_cache_dir=roi_cache_dir,
        roi_cache_manifest=roi_cache_manifest,
        roi_cache_expected_archive_path=roi_cache_expected_archive_path,
        source_crop_row_start=source_crop_row_start,
        source_crop_row_stop=source_crop_row_stop,
        stage_roi_cache_to_scratch=bool(stage_roi_cache_to_scratch),
        roi_cache_staging_dir=roi_cache_staging_dir,
        profile_timings=bool(profile_timings),
        progress_jsonl=progress_jsonl,
        progress_every_batches=int(progress_every_batches),
        input_mode=input_mode,
        model_input_transform_mode=model_input_transform_mode,
        coordinate_contract_mode=coordinate_contract_mode,
        require_training_materialization_binding=bool(
            require_training_materialization_binding
        ),
        keypoint_roi_shard_rows=keypoint_roi_shard_rows,
        keypoint_frame_shard_rows=int(keypoint_frame_shard_rows),
        cpu=bool(cpu),
        verbose=bool(verbose),
    )

    candidate_binding_rejections: list[dict[str, str]] = []
    compatible_bindings: dict[str, dict[str, Any]] = {}
    registry_db: Optional[Registry] = None
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
            keypoint_parent=output_parent,
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
        if model_run_id is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.run_id == str(model_run_id)
            ]
        candidates, compatible_bindings, candidate_binding_rejections = (
            validate_pose_candidate_bindings(
                registry_db,
                candidates,
                expected_pose_schema=pose_schema,
            )
        )
    except Exception as exc:
        return _failure_result(
            reason="model_resolution_failed",
            error=str(exc),
            remediation="Verify registry metadata for this recording and rerun with --include-non-success or --set-id as needed.",
            recording_dir=resolved_recording_dir,
            output_path=output_path,
            registry_path=registry_path,
            keypoint_parent=output_parent,
        )
    finally:
        if registry_db is not None:
            registry_db.close()

    if not candidates and candidate_binding_rejections:
        return _failure_result(
            reason="model_pose_schema_binding_failed",
            error=(
                "No registry pose candidate has a valid hash-bound skeleton binding: "
                + json.dumps(candidate_binding_rejections, sort_keys=True)
            ),
            remediation=(
                "Repair or replace the rejected registry model metadata. Automatic "
                "resolution will not rank schema-invalid pose candidates."
            ),
            recording_dir=resolved_recording_dir,
            output_path=output_path,
            registry_path=registry_path,
            keypoint_parent=output_parent,
        )

    try:
        best = pick_best_keypoint_candidate(candidates, require_unique=bool(require_unique))
    except SystemExit as exc:
        return _failure_result(
            reason="candidate_selection_failed",
            error=str(exc),
            remediation=(
                "Verify --model-run-id and --set-id identify a successful pose model, "
                "or remove --require-unique."
            ),
            recording_dir=resolved_recording_dir,
            output_path=output_path,
            registry_path=registry_path,
            keypoint_parent=output_parent,
        )

    try:
        model_pose_schema_binding = compatible_bindings[best.run_id]
    except KeyError:
        return _failure_result(
            reason="model_pose_schema_binding_failed",
            error=f"Validated pose binding disappeared for selected run {best.run_id!r}.",
            remediation=(
                "Repair or replace the selected registry model's hash-bound training "
                "manifest/skeleton metadata; canonical inference will not infer an "
                "ordered keypoint axis from cardinality or defaults."
            ),
            recording_dir=resolved_recording_dir,
            output_path=output_path,
            registry_path=registry_path,
            keypoint_parent=output_parent,
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
    payload.setdefault("artifacts", {})["model_pose_schema_binding"] = (
        model_pose_schema_binding
    )
    payload.setdefault("artifacts", {})["rejected_pose_candidates"] = (
        candidate_binding_rejections
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
            keypoint_parent=output_parent,
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
                    "output_parent": output_parent,
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
            expected_model_stride=expected_model_stride,
            run_name=run_name,
            output_parent=output_parent,
            crop_run=crop_run,
            pose_schema=pose_schema,
            model_pose_schema_binding=model_pose_schema_binding,
            batch_size=batch_size,
            device=resolved_device,
            imgsz=imgsz,
            model_input_size=model_input_size,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=bool(verbose),
            mask_threshold=mask_threshold,
            roi_cache_policy=roi_cache_policy,
            roi_cache_dir=roi_cache_dir,
            roi_cache_manifest=effective_roi_cache_manifest,
            roi_cache_expected_archive_path=roi_cache_expected_archive_path,
            source_crop_row_start=source_crop_row_start,
            source_crop_row_stop=source_crop_row_stop,
            roi_cache_source_tier=roi_cache_staging_details.get("effective_source_tier"),
            roi_cache_staged_to_node_scratch=bool(roi_cache_staging_details.get("staged", False)),
            roi_cache_staging_details=roi_cache_staging_details or None,
            input_mode=input_mode,
            model_input_transform_mode=model_input_transform_mode,
            coordinate_contract_mode=coordinate_contract_mode,
            require_training_materialization_binding=bool(
                require_training_materialization_binding
            ),
            keypoint_roi_shard_rows=keypoint_roi_shard_rows,
            keypoint_frame_shard_rows=int(keypoint_frame_shard_rows),
            profile_timings=bool(profile_timings),
            progress_jsonl=progress_jsonl,
            progress_every_batches=progress_every_batches,
            registry=registry_path,
            cli_provenance=effective_run_provenance,
            run_provenance=effective_run_provenance,
        )
        if not keypoint_run:
            raise RuntimeError("Keypoint inference did not create a run; model resolution provenance cannot be written.")
        # Canonical publication is immutable after completion. Its selected
        # registry run/set are already in run provenance and the exact
        # model-schema binding is digest-sealed in the coordinate context.
        # Legacy collection shards may retain the historical post-write annotation.
        if output_parent != DEFAULT_KEYPOINT_OUTPUT_PARENT:
            write_keypoint_model_resolution_provenance(
                zarr_path=output_path,
                run_name=keypoint_run,
                payload=payload,
                output_parent=output_parent,
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
            keypoint_parent=output_parent,
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
        keypoint_parent=output_parent,
        resolved_at_utc=resolved_at_utc,
        resolution_payload=payload,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True, help="Recording directory to process.")
    parser.add_argument("--output", type=Path, help="Optional explicit output zarr path.")
    parser.add_argument("--registry", type=Path, help="Optional registry sqlite path.")
    parser.add_argument("--set-id", type=str, help="Optional set filter during model resolution.")
    parser.add_argument(
        "--model-run-id",
        type=str,
        help="Optional exact registered pose training run to select.",
    )
    parser.add_argument("--require-unique", action="store_true", help="Fail if top scores tie.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to store in provenance.")
    parser.add_argument("--include-non-success", action="store_true", help="Include non-success training runs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve model only; do not run keypoints.")

    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit keypoints run name.")
    parser.add_argument(
        "--output-parent",
        choices=KEYPOINT_OUTPUT_PARENTS,
        default=DEFAULT_KEYPOINT_OUTPUT_PARENT,
        help=(
            "Parent group for the output run. Use keypoint_shard_runs for clipped-collection "
            "model shards that will later be finalized into keypoints_runs."
        ),
    )
    parser.add_argument("--crop-run", type=str, default=None, help="Optional explicit crop run name.")
    parser.add_argument(
        "--pose-schema",
        type=str,
        default=None,
        help=(
            "Optional package-schema consistency assertion. Canonical ordered "
            "labels are resolved from the selected model's hash-bound training manifest."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Optional keypoint batch size override.")
    keypoint_storage_group = parser.add_mutually_exclusive_group()
    keypoint_storage_group.add_argument(
        "--keypoint-roi-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
        help=(
            "Requested outer rows for indexed-sharded ROI-domain YOLO keypoint arrays "
            f"(default: {DEFAULT_KEYPOINT_ROI_SHARD_ROWS})."
        ),
    )
    keypoint_storage_group.add_argument(
        "--no-keypoint-sharding",
        action="store_const",
        dest="keypoint_roi_shard_rows",
        const=None,
        help="Use ordinary chunks for YOLO keypoint outputs.",
    )
    parser.add_argument(
        "--keypoint-frame-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
        help=(
            "Aligned outer shard rows for frame-domain arrays when keypoint sharding is enabled "
            f"(default: {DEFAULT_KEYPOINT_FRAME_SHARD_ROWS})."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional Ultralytics network preprocessing size override.",
    )
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=None,
        help=(
            "Optional square pixel extent submitted before Ultralytics internal "
            "preprocessing; defaults to --imgsz."
        ),
    )
    parser.add_argument(
        "--expected-model-stride",
        type=int,
        default=None,
        help="Fail unless the loaded pose model declares this maximum stride.",
    )
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
        "--roi-cache-expected-archive-path",
        type=Path,
        default=None,
        help=(
            "Original analysis archive bound by a cache staged into a local "
            "compute shell."
        ),
    )
    parser.add_argument("--source-crop-row-start", type=int, default=None)
    parser.add_argument("--source-crop-row-stop", type=int, default=None)
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
    parser.add_argument(
        "--model-input-transform",
        choices=MODEL_INPUT_TRANSFORM_CHOICES,
        default="auto",
        help="Exact reversible native-ROI to model-input transform.",
    )
    parser.add_argument(
        "--coordinate-contract-mode",
        choices=("canonical", "legacy_noncanonical"),
        default="canonical",
        help=(
            "Canonical publication mode, or explicit noncanonical terminal "
            "compute output for a later strict v2 finalizer."
        ),
    )
    parser.add_argument(
        "--require-training-materialization-binding",
        action="store_true",
        help=(
            "Require the explicit training crop-materialization v1 binding and "
            "write only a non-authoritative terminal keypoint shard."
        ),
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
        model_run_id=args.model_run_id,
        require_unique=bool(args.require_unique),
        top_k=int(args.top_k),
        include_non_success=bool(args.include_non_success),
        dry_run=bool(args.dry_run),
        run_name=args.run_name,
        output_parent=args.output_parent,
        crop_run=args.crop_run,
        pose_schema=args.pose_schema,
        batch_size=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        model_input_size=args.model_input_size,
        expected_model_stride=args.expected_model_stride,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        mask_threshold=args.mask_threshold,
        roi_cache_policy=args.roi_cache_policy,
        roi_cache_dir=args.roi_cache_dir,
        roi_cache_manifest=args.roi_cache_manifest,
        roi_cache_expected_archive_path=args.roi_cache_expected_archive_path,
        source_crop_row_start=args.source_crop_row_start,
        source_crop_row_stop=args.source_crop_row_stop,
        stage_roi_cache_to_scratch=bool(args.stage_roi_cache_to_scratch),
        roi_cache_staging_dir=args.roi_cache_staging_dir,
        profile_timings=bool(args.profile_timings),
        progress_jsonl=args.progress_jsonl,
        progress_every_batches=args.progress_every_batches,
        input_mode=args.input_mode,
        model_input_transform_mode=args.model_input_transform,
        coordinate_contract_mode=args.coordinate_contract_mode,
        require_training_materialization_binding=bool(
            args.require_training_materialization_binding
        ),
        keypoint_roi_shard_rows=args.keypoint_roi_shard_rows,
        keypoint_frame_shard_rows=args.keypoint_frame_shard_rows,
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
    print(f"  keypoint_parent: {result.keypoint_parent}")
    print(f"  keypoint_run: {result.keypoint_run}")
    print(f"  selected_model: {result.selected_model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
