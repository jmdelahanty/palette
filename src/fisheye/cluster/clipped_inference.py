"""Plan registry-pinned clipped-recording workflow scopes on LSF."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.cluster.arena_geometry import (
    RegisteredDetectionGateFragmentInputs,
    build_registered_detection_gate_fragment,
)
from fisheye.cluster.clipped_detection import (
    DetectionFragmentInputs,
    DetectionFragmentOutputs,
    DetectionModelSpec,
    DetectionWorkflowModule,
    DetectionWorkUnitSpec,
    RawDetectionFragmentOutputs,
    RawDetectionWorkflowModule,
    build_detection_fragment,
)
from fisheye.cluster.clipped_detection_evidence import (
    ClipDetectionEvidenceInput,
    ClippedDetectionEvidenceInputs,
    build_clipped_detection_storage_fragments,
)
from fisheye.cluster.clipped_lsf import (
    build_execution_task as _execution_task,
    build_job as _job,
    build_task_group_job as _task_group_job,
    chain_commands as _chain,
)
from fisheye.cluster.keypoints.common import (
    resolve_pose_model_binding,
    safe_component,
    validate_registered_analysis_zarr,
)
from fisheye.cluster.lsf import (
    CommandRunner,
    LsfExecutionMode,
    LsfExecutionTask,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
    shell_join,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_JOB_INDEX_TOKEN,
    RUNTIME_USER_TOKEN,
)
from fisheye.cluster.recording_layout import clipped_recording_target
from fisheye.cluster.recording_detection_postprocess import (
    REGISTERED_GATE_REQUIREMENTS,
    RecordingDetectionPostprocessInputs,
    build_recording_detection_postprocess_fragment,
)
from fisheye.cluster.clipped_storage_finalization import (
    ClippedStorageFinalizationInputs,
    StrictClipRefinedDetectionInput,
)
from fisheye.cluster.native_detection import (
    NativeDetectionClipSpec,
    NativeDetectionFragmentInputs,
    NativeDetectionModelSpec,
    build_native_detection_fragment,
)
from fisheye.cluster.native_detection_authority import (
    load_native_archive_authority,
    recording_frame_work_unit_intervals,
    validate_recording_frame_index,
)
from fisheye.registry.db import Registry
from fisheye.registry.model_resolution import (
    load_candidates,
    load_subject_mask_model_candidates,
    load_target_profile,
    resolve_recording_id,
    verify_deployment_artifact_content,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.utils.plan_clipped_detect_refine_workflow import (
    build_plan as build_detection_plan,
)
from fisheye.utils.validate_imported_run_group import validate_imported_run_group

LEGACY_PLAN_SCHEMA = "palette.clipped_inference_bsub_plan.v1"
PLAN_SCHEMA = "palette.clipped_inference_bsub_plan.v2"
SUPPORTED_PLAN_SCHEMAS = frozenset((LEGACY_PLAN_SCHEMA, PLAN_SCHEMA))
TARGET_MANIFEST_SCHEMA = "palette.clipped_inference_targets.v1"
FAMILY = "clipped_inference"
WORKFLOW_SCOPE_FULL = "full"
WORKFLOW_SCOPE_DETECTION = "detection"
WORKFLOW_SCOPE_DOWNSTREAM = "downstream"
WORKFLOW_SCOPES = (
    WORKFLOW_SCOPE_FULL,
    WORKFLOW_SCOPE_DETECTION,
    WORKFLOW_SCOPE_DOWNSTREAM,
)
DEFAULT_REPO = Path("/groups/johnson/johnsonlab/jeremy/gitrepos/palette")
DEFAULT_REGISTRY = Path(
    "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
)
DEFAULT_CACHE_ROOT = Path("/nrs/johnson/palette_staging/flat_roi_cache")
DEFAULT_PACKAGE_ROOT = Path(
    "/nrs/johnson/palette_staging/refined_subject_mask_clip_packages"
)
SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED = "receipt_composed_v1"
SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK = "streaming_rollback_v1"
SUBJECT_MASK_PUBLICATION_PROFILES = (
    SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED,
    SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK,
)


@dataclass(frozen=True)
class CampaignTarget:
    target_id: str
    recording_id: str
    recording_dir: Path
    analysis_zarr: Path
    expected_subject_count: int = 1
    finalized_refined_detect_run: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "recording_id": self.recording_id,
            "recording_dir": str(self.recording_dir),
            "analysis_zarr": str(self.analysis_zarr),
            "expected_subject_count": int(self.expected_subject_count),
            "finalized_refined_detect_run": self.finalized_refined_detect_run,
        }


@dataclass(frozen=True)
class ModelBinding:
    task: str
    set_id: str
    run_id: str
    path: Path
    sha256: str

    def to_json(self) -> dict[str, str]:
        return {
            "task": self.task,
            "set_id": self.set_id,
            "run_id": self.run_id,
            "path": str(self.path),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ClippedInferencePlan:
    run_label: str
    workflow_id: str
    workflow_scope: str
    repo: Path
    palette_commit: str
    registry: Path
    run_root: Path
    targets: tuple[CampaignTarget, ...]
    target_plans: tuple[Mapping[str, Any], ...]
    model_bindings: Mapping[str, ModelBinding]
    max_active_targets: int
    cleanup_nrs_after_success: bool
    resume_existing_detections: bool
    encoded_mask_packages: bool
    subject_mask_publication_profile: str
    detect_array_concurrency: int
    gpu_array_concurrency: int
    cache_array_concurrency: int
    mask_package_array_concurrency: int
    detect_refine_bundle_concurrency: int
    registered_gate_requirement: str
    registered_gate_run: str | None
    selection_policy_id: str
    lsf_workflow: LsfWorkflow

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "run_label": self.run_label,
            "workflow_id": self.workflow_id,
            "workflow_scope": self.workflow_scope,
            "repo": str(self.repo),
            "palette_commit": self.palette_commit,
            "registry": str(self.registry),
            "run_root": str(self.run_root),
            "target_count": len(self.targets),
            "targets": list(self.target_plans),
            "models": {
                name: binding.to_json() for name, binding in self.model_bindings.items()
            },
            "max_active_targets": self.max_active_targets,
            "cleanup_nrs_after_success": self.cleanup_nrs_after_success,
            "resume_existing_detections": self.resume_existing_detections,
            "encoded_mask_packages": self.encoded_mask_packages,
            "subject_mask_publication_profile": self.subject_mask_publication_profile,
            "registered_dish_geometry": {
                "gate_requirement": self.registered_gate_requirement,
                "gate_run": self.registered_gate_run,
                "selection_policy_id": self.selection_policy_id,
            },
            "scheduler_concurrency": {
                "detect_array": self.detect_array_concurrency,
                "gpu_array_per_stage": self.gpu_array_concurrency,
                "cache_array": self.cache_array_concurrency,
                "mask_package_array": self.mask_package_array_concurrency,
                "detect_refine_bundle": self.detect_refine_bundle_concurrency,
            },
            "lsf_workflow": self.lsf_workflow.to_json(),
        }


def load_target_manifest(path: Path) -> tuple[CampaignTarget, ...]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema") != TARGET_MANIFEST_SCHEMA:
        raise ValueError(f"Target manifest schema must be {TARGET_MANIFEST_SCHEMA!r}.")
    rows = payload.get("targets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Target manifest requires a non-empty targets list.")
    targets: list[CampaignTarget] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Target row {index} is not an object.")
        recording_id = str(row.get("recording_id") or "").strip()
        recording_dir = Path(str(row.get("recording_dir") or "")).expanduser().resolve()
        analysis_zarr = Path(str(row.get("analysis_zarr") or "")).expanduser().resolve()
        target_id = safe_component(
            str(row.get("target_id") or recording_dir.name),
            default=f"target_{index:03d}",
            max_length=80,
        )
        if not recording_id:
            raise ValueError(f"Target {target_id!r} has no recording_id.")
        raw_expected_subject_count = row.get("expected_subject_count")
        expected_subject_count = int(
            1 if raw_expected_subject_count is None else raw_expected_subject_count
        )
        if expected_subject_count <= 0:
            raise ValueError(
                f"Target {target_id!r} expected_subject_count must be positive."
            )
        if not (recording_dir / "recording_clip_index.json").is_file():
            raise FileNotFoundError(
                f"Target {target_id!r} has no recording_clip_index.json: {recording_dir}"
            )
        if not (analysis_zarr / "zarr.json").is_file():
            raise FileNotFoundError(
                f"Target {target_id!r} is not a Zarr v3 root: {analysis_zarr}"
            )
        targets.append(
            CampaignTarget(
                target_id=target_id,
                recording_id=recording_id,
                recording_dir=recording_dir,
                analysis_zarr=analysis_zarr,
                expected_subject_count=expected_subject_count,
                finalized_refined_detect_run=(
                    str(row.get("finalized_refined_detect_run") or "").strip() or None
                ),
            )
        )
    if len({target.target_id for target in targets}) != len(targets):
        raise ValueError("Target ids must be unique.")
    if len({target.analysis_zarr for target in targets}) != len(targets):
        raise ValueError("Analysis Zarr targets must be unique.")
    return tuple(targets)


def _verify_binding(binding: ModelBinding) -> None:
    verification = verify_deployment_artifact_content(
        {"model_path": str(binding.path), "model_sha256": binding.sha256},
        artifact="model",
        role=f"{binding.task}_deployment_model",
    )
    if verification.status != "match":
        raise ValueError(
            f"Registered {binding.task} model is not content-pinned: "
            f"{json.dumps(verification.to_dict(), sort_keys=True)}"
        )


def _resolve_ranked_binding(
    *,
    registry_path: Path,
    target: CampaignTarget,
    task: str,
    set_id: str,
    run_id: str,
) -> ModelBinding:
    registry = Registry(registry_path)
    try:
        resolved_id = resolve_recording_id(
            registry,
            recording_id=target.recording_id,
            recording_dir=target.recording_dir,
        )
        profile = load_target_profile(registry, resolved_id)
        candidates = load_candidates(
            registry,
            target=profile,
            task=task,
            set_id_filter=set_id,
            include_non_success=False,
        )
        exact = [candidate for candidate in candidates if candidate.run_id == run_id]
    finally:
        registry.close()
    if len(exact) != 1:
        raise ValueError(
            f"Expected one successful {task} model for set {set_id!r}, run {run_id!r}; "
            f"found {len(exact)} for {target.recording_id!r}."
        )
    candidate = exact[0]
    if not candidate.model_sha256:
        raise ValueError(f"Registered {task} model {run_id!r} has no model_sha256.")
    binding = ModelBinding(
        task=task,
        set_id=candidate.set_id,
        run_id=candidate.run_id,
        path=Path(candidate.model_path).expanduser().resolve(),
        sha256=candidate.model_sha256,
    )
    _verify_binding(binding)
    return binding


def resolve_detection_model_binding(
    *,
    registry_path: Path,
    target: CampaignTarget,
    set_id: str,
    run_id: str,
) -> ModelBinding:
    """Resolve and content-verify one exact registered detection model."""

    return _resolve_ranked_binding(
        registry_path=registry_path,
        target=target,
        task="detect",
        set_id=set_id,
        run_id=run_id,
    )


def _resolve_subject_binding(
    *,
    registry_path: Path,
    set_id: str,
    run_id: str,
    coverage_class: str,
    component_coverage_key: str,
    label_schema_id: str,
) -> ModelBinding:
    registry = Registry(registry_path)
    try:
        candidates = load_subject_mask_model_candidates(
            registry,
            set_id=set_id,
            run_id=run_id,
            coverage_class=coverage_class,
            component_coverage_key=component_coverage_key,
            label_schema_id=label_schema_id,
            include_non_success=False,
            require_existing_path=True,
        )
    finally:
        registry.close()
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one successful subject-mask model for set {set_id!r}, "
            f"run {run_id!r}; found {len(candidates)}."
        )
    candidate = candidates[0]
    if not candidate.model_sha256:
        raise ValueError(
            f"Registered subject-mask model {run_id!r} has no model_sha256."
        )
    binding = ModelBinding(
        task="subject_masks",
        set_id=str(candidate.set_id),
        run_id=candidate.run_id,
        path=Path(candidate.model_path).expanduser().resolve(),
        sha256=candidate.model_sha256,
    )
    _verify_binding(binding)
    return binding


def _assert_same_binding(name: str, bindings: Sequence[ModelBinding]) -> ModelBinding:
    if not bindings:
        raise ValueError(f"No {name} model bindings were resolved.")
    identities = {
        (item.set_id, item.run_id, item.path, item.sha256) for item in bindings
    }
    if len(identities) != 1:
        raise ValueError(
            f"The target cohort does not resolve one common {name} model: {identities}"
        )
    return bindings[0]


def _refuse_output_collisions(
    target_plan: Mapping[str, Any],
    *,
    workflow_scope: str,
    allow_existing_detections: bool = False,
) -> None:
    zarr = Path(str(target_plan["analysis_zarr"]))
    outputs: list[Path] = []
    if workflow_scope == WORKFLOW_SCOPE_DOWNSTREAM:
        for clip in target_plan["clips"]:
            outputs.extend(
                [
                    zarr / "keypoint_shard_runs" / str(clip["keypoint_shard_run"]),
                    zarr
                    / "subject_mask_shard_runs"
                    / str(clip["subject_mask_shard_run"]),
                ]
            )
        outputs.extend(
            [
                zarr / "crop_runs" / str(target_plan["hybrid_crop_run"]),
                zarr / "keypoints_runs" / str(target_plan["keypoint_run"]),
                zarr
                / "refined_keypoints_runs"
                / str(target_plan["refined_keypoint_run"]),
                zarr / "subject_mask_runs" / str(target_plan["subject_mask_run"]),
                zarr
                / "refined_subject_masks_runs"
                / str(target_plan["refined_subject_mask_run"]),
                zarr
                / "subject_mask_quality_runs"
                / str(target_plan["subject_mask_quality_run"]),
                zarr
                / "subject_mask_cache_runs"
                / str(target_plan["subject_mask_cache_run"]),
                zarr
                / "subject_mask_bundle_runs"
                / str(target_plan["subject_mask_bundle_id"]),
                Path(str(target_plan["hybrid_supplemental_manifest"])),
                Path(str(target_plan["package_dir"])),
            ]
        )
        collisions = [path for path in outputs if path.exists()]
        if collisions:
            raise FileExistsError(
                "Planned immutable outputs already exist: "
                + ", ".join(str(path) for path in collisions)
            )
        return
    for clip in target_plan["clips"]:
        if not allow_existing_detections:
            outputs.append(zarr / str(clip["detect_group_path"]))
        if not target_plan.get("canonical_refined_run_id"):
            outputs.append(zarr / str(clip["refined_detect_group_path"]))
    outputs.extend(
        [
            zarr
            / "experiment_index"
            / "finalized_runs"
            / str(target_plan["collection_id"]),
            zarr
            / "detect_collection_sources"
            / str(target_plan["detect_quality_source_run"]),
            zarr / "detect_quality_runs" / str(target_plan["detect_quality_run"]),
        ]
    )
    canonical_run = str(target_plan.get("native_canonical_run_id") or "")
    if canonical_run:
        outputs.append(zarr / "detect_runs" / canonical_run)
    canonical_refined_run = str(target_plan.get("canonical_refined_run_id") or "")
    if canonical_refined_run:
        outputs.append(zarr / "refined_detect_runs" / canonical_refined_run)
    planned_gate_group = str(
        target_plan.get("planned_registered_gate_group_path") or ""
    )
    if planned_gate_group:
        outputs.append(zarr / planned_gate_group)
    if workflow_scope == WORKFLOW_SCOPE_FULL:
        for clip in target_plan["clips"]:
            outputs.extend(
                [
                    zarr / "crop_runs" / str(clip["proxy_crop_run"]),
                    zarr / "keypoint_shard_runs" / str(clip["keypoint_shard_run"]),
                    zarr
                    / "subject_mask_shard_runs"
                    / str(clip["subject_mask_shard_run"]),
                ]
            )
        outputs.extend(
            [
                zarr / "crop_runs" / str(target_plan["merged_proxy_crop_run"]),
                zarr / "keypoints_runs" / str(target_plan["keypoint_run"]),
                zarr
                / "refined_keypoints_runs"
                / str(target_plan["refined_keypoint_run"]),
                zarr / "subject_mask_runs" / str(target_plan["subject_mask_run"]),
                zarr
                / "refined_subject_masks_runs"
                / str(target_plan["refined_subject_mask_draft_run"]),
                zarr
                / "refined_subject_masks_runs"
                / str(target_plan["refined_subject_mask_run"]),
                zarr
                / "subject_mask_quality_runs"
                / str(target_plan["subject_mask_quality_run"]),
                zarr
                / "subject_mask_cache_runs"
                / str(target_plan["subject_mask_cache_run"]),
                zarr
                / "subject_mask_bundle_runs"
                / str(target_plan["subject_mask_bundle_id"]),
                Path(str(target_plan["cache_dir"])),
                Path(str(target_plan["package_dir"])),
            ]
        )
        strict_bundle = str(target_plan.get("strict_storage_bundle_root") or "")
        if strict_bundle:
            outputs.append(Path(strict_bundle))
    collisions = [path for path in outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "Planned immutable outputs already exist: "
            + ", ".join(str(path) for path in collisions)
        )


def _planned_crop_row_count(clips: Sequence[Mapping[str, Any]]) -> int:
    """Require clip row intervals to form one exact recording-row partition."""

    expected_start = 0
    for position, clip in enumerate(clips):
        clip_id = str(clip.get("clip_id") or f"clip[{position}]")
        row_start = int(clip["crop_row_start"])
        row_stop = int(clip["crop_row_stop"])
        if row_stop < row_start:
            raise ValueError(
                f"Clip {clip_id!r} has reversed crop row interval "
                f"[{row_start}, {row_stop})."
            )
        if row_start != expected_start:
            raise ValueError(
                "Clip crop row intervals must be contiguous, non-overlapping, and "
                f"recording-ordered; clip {clip_id!r} starts at {row_start}, "
                f"expected {expected_start}."
            )
        expected_start = row_stop
    return expected_start


def _read_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _clipped_layout_work_units(
    target: CampaignTarget,
    *,
    camera_serial: str,
) -> list[dict[str, Any]]:
    payload = _read_strict_json(target.recording_dir / "recording_clip_index.json")
    raw_rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(raw_rows, list):
        raise ValueError("recording_clip_index.json has no rows array.")
    work_units: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("camera_serial") or "").strip() != str(camera_serial):
            continue
        clip_index = int(raw.get("clip_index", -1))
        clip_id = str(raw.get("clip_id") or f"clip_{clip_index:06d}")
        video_value = raw.get("video_path") or raw.get("video")
        video_path = Path(str(video_value or "")).expanduser()
        if not video_path.is_absolute():
            video_path = target.recording_dir / video_path
        video_path = video_path.resolve()
        if clip_index < 0 or not video_path.is_file():
            raise ValueError(f"Invalid clipped work unit {clip_id!r}.")
        work_units.append(
            {
                "work_unit_id": f"{target.recording_id}_{clip_id}_cam{camera_serial}",
                "clip_id": clip_id,
                "clip_index": clip_index,
                "camera_serial": str(camera_serial),
                "frame_count": int(raw.get("frame_count")),
                "source": {"video_path": str(video_path)},
            }
        )
    work_units.sort(key=lambda unit: int(unit["clip_index"]))
    if not work_units:
        raise ValueError(f"No clipped work units found for camera {camera_serial}.")
    return work_units


def _refined_clip_crop_row_intervals(
    *,
    target: CampaignTarget,
    refined_run_name: str,
    frame_intervals: Mapping[tuple[int, str], tuple[int, int]],
) -> dict[tuple[int, str], tuple[int, int]]:
    bound = bind_refined_detection_crop_source(
        target.analysis_zarr,
        run_id=refined_run_name,
        allow_selector_ineligible_benchmark=True,
    )
    attrs = bound.run_group.attrs
    gate = attrs.get("registered_detection_gate")
    if (
        bound.run_id != refined_run_name
        or attrs.get("status") != "complete"
        or attrs.get("finalized_recording_authority") is not True
        or attrs.get("immutable_snapshot") is not True
        or attrs.get("registered_detection_gate_requirement") != "required"
        or not isinstance(gate, Mapping)
        or gate.get("status") != "applied"
        or gate.get("applied") is not True
        or gate.get("ordered_instance_key_coverage_exact") is not True
    ):
        raise ValueError(
            f"refined_detect_runs/{refined_run_name} is not a complete gated "
            "immutable recording authority."
        )
    instances = bound.instances_group
    frames = np.asarray(instances["frame_indices"][:], dtype=np.int64).reshape(-1)
    acquisition_frames = np.asarray(
        instances["source_acquisition_frame_index"][:], dtype=np.int64
    ).reshape(-1)
    if not np.array_equal(frames, acquisition_frames):
        raise ValueError(
            "Finalized refined rows disagree with acquisition-frame identity."
        )
    if frames.size and np.any(frames[1:] < frames[:-1]):
        raise ValueError("Finalized refined detection rows are not recording ordered.")
    intervals: dict[tuple[int, str], tuple[int, int]] = {}
    cursor = 0
    for key, (frame_start, frame_stop) in sorted(
        frame_intervals.items(), key=lambda item: item[1][0]
    ):
        row_start = int(np.searchsorted(frames, int(frame_start), side="left"))
        row_stop = int(np.searchsorted(frames, int(frame_stop), side="left"))
        if row_start != cursor:
            raise ValueError("Refined detection clip partitions are not contiguous.")
        intervals[key] = (row_start, row_stop)
        cursor = row_stop
    if cursor != int(frames.shape[0]):
        raise ValueError(
            "Refined detection rows extend outside clipped frame intervals."
        )
    return intervals


def _active_arena_geometry_selection(analysis_zarr: Path) -> str:
    parent = analysis_zarr / "analysis" / "arena_geometry_selection"
    metadata = _read_strict_json(parent / "zarr.json")
    attrs = metadata.get("attributes")
    if not isinstance(attrs, Mapping):
        raise ValueError("Arena geometry selection parent has no attributes.")
    latest = str(attrs.get("latest") or "").strip()
    latest_complete = str(attrs.get("latest_complete") or "").strip()
    if not latest or latest != latest_complete:
        raise ValueError(
            "Required registered geometry needs one active complete selection."
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", latest):
        raise ValueError("Active arena geometry selection has an unsafe run name.")
    run_metadata = _read_strict_json(parent / latest / "zarr.json")
    run_attrs = run_metadata.get("attributes")
    if not isinstance(run_attrs, Mapping):
        raise ValueError("Active arena geometry selection has no attributes.")
    if run_attrs.get("schema_id") != "palette.arena_geometry_selection_run":
        raise ValueError("Active arena geometry selection has an unsupported schema.")
    if str(run_attrs.get("selection_id") or "") != latest:
        raise ValueError("Active arena geometry selection identity disagrees.")
    return latest


@dataclass(frozen=True)
class _DownstreamTargetPipeline:
    jobs: tuple[LsfJob, ...]
    fragments: tuple[LsfWorkflowFragment, ...]
    terminal_job_key: str
    terminal_artifact_key: str


def _build_downstream_target_pipeline(
    *,
    workflow_id: str,
    repo: Path,
    repo_commit: str,
    registry_path: Path,
    run_root: Path,
    target: CampaignTarget,
    target_safe: str,
    target_payload: dict[str, Any],
    pose_binding: ModelBinding,
    subject_binding: ModelBinding,
    subject_mask_coverage_class: str,
    subject_mask_component_coverage_key: str,
    subject_mask_label_schema_id: str,
    subject_mask_publication_profile: str,
    encoded_mask_packages: bool,
    gpu_array_concurrency: int,
    mask_package_array_concurrency: int,
    keypoint_gpu_queue: str,
    subject_mask_gpu_queue: str,
    cpu: LsfResources,
    final_cpu: LsfResources,
    import_cpu: LsfResources,
    cache_gpu: LsfResources,
    upstream_job_keys: tuple[str, ...],
    required_artifacts: tuple[str, ...],
) -> _DownstreamTargetPipeline:
    """Compose downstream analysis from one exact recording-level refined run."""

    if target.finalized_refined_detect_run is None:
        raise ValueError("Downstream target has no finalized refined-detection run.")
    clips = list(target_payload["clips"])
    hybrid_crop_run = str(target_payload["hybrid_crop_run"])
    keypoint_run = str(target_payload["keypoint_run"])
    refined_keypoint_run = str(target_payload["refined_keypoint_run"])
    subject_mask_run = str(target_payload["subject_mask_run"])
    refined_subject_mask_run = str(target_payload["refined_subject_mask_run"])
    refined_subject_mask_draft_run = str(
        target_payload["refined_subject_mask_draft_run"]
    )
    subject_mask_quality_run = str(target_payload["subject_mask_quality_run"])
    subject_mask_cache_run = str(target_payload["subject_mask_cache_run"])
    subject_mask_bundle_id = str(target_payload["subject_mask_bundle_id"])
    global_mask_grid_manifest = Path(str(target_payload["global_mask_grid_manifest"]))
    supplemental_manifest = Path(str(target_payload["hybrid_supplemental_manifest"]))
    source_run_path = f"refined_detect_runs/{target.finalized_refined_detect_run}"

    jobs: list[LsfJob] = []
    fragments: list[LsfWorkflowFragment] = []
    ledger_key = f"acquisition_crop_ledger:{target_safe}"
    ledger_report = run_root / "ledger" / f"{target_safe}.jsonl"
    ledger_job = _job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=ledger_key,
        stage="acquisition_crop_ledger",
        command=(
            "scripts/py",
            "-m",
            "fisheye.utils.backfill_acquisition_video_stream_inventory",
            str(target.recording_dir),
            "--apply",
            "--output-jsonl",
            str(ledger_report),
        ),
        resources=cpu,
        upstream=upstream_job_keys,
        expected_outputs=(ledger_report,),
    )
    jobs.append(ledger_job)
    ledger_artifact = f"acquisition_crop_ledger:{target_safe}"
    fragments.append(
        LsfWorkflowFragment(
            fragment_id=f"acquisition_crop_ledger:{target_safe}",
            jobs=(ledger_job,),
            requires=required_artifacts,
            provides=(ledger_artifact,),
            metadata={
                "module": "acquisition_crop_ledger",
                "target_id": target.target_id,
                "recording_layout": "rolling_clip_collection_v1",
                "publication_authority": "one_recording_level_canonical_ledger",
            },
        )
    )

    hybrid_key = f"hybrid_crop:{target_safe}"
    hybrid_report = run_root / "hybrid_crop" / f"{target_safe}.json"
    hybrid_job = _job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=hybrid_key,
        stage="hybrid_crop_provider",
        command=(
            "scripts/py",
            "-m",
            "fisheye.utils.build_hybrid_acquisition_offline_crop_run",
            str(target.analysis_zarr),
            "--refined-detect-run",
            target.finalized_refined_detect_run,
            "--run-name",
            hybrid_crop_run,
            "--recording-dir",
            str(target.recording_dir),
            "--supplemental-manifest-path",
            str(supplemental_manifest),
            "--decode-mode",
            "indexed",
            "--apply",
            "--output-json",
            str(hybrid_report),
        ),
        resources=cache_gpu,
        upstream=(ledger_key,),
        expected_outputs=(
            target.analysis_zarr / "crop_runs" / hybrid_crop_run / "zarr.json",
            hybrid_report,
        ),
    )
    jobs.append(hybrid_job)
    hybrid_artifact = f"hybrid_crop_provider:{target_safe}"
    fragments.append(
        LsfWorkflowFragment(
            fragment_id=f"hybrid_crop_provider:{target_safe}",
            jobs=(hybrid_job,),
            requires=(ledger_artifact,),
            provides=(hybrid_artifact,),
            metadata={
                "module": "hybrid_crop_provider",
                "target_id": target.target_id,
                "source_refined_detect_run": target.finalized_refined_detect_run,
                "crop_run": hybrid_crop_run,
                "crop_authority": "one_signed_recording_level_provider",
                "selector_activation": False,
            },
        )
    )

    keypoint_array_key = f"keypoints_array:{target_safe}"
    subject_mask_array_key = f"subject_masks_array:{target_safe}"
    keypoint_preflight_key = f"keypoint_finalize_preflight:{target_safe}"
    expected_target_crop_rows = _planned_crop_row_count(clips)
    keypoint_preflight_job = _job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=keypoint_preflight_key,
        stage="keypoint_finalize_preflight",
        command=(
            "scripts/py",
            "-m",
            "fisheye.utils.finalize_keypoint_shards",
            str(target.analysis_zarr),
            "--target-crop-run",
            hybrid_crop_run,
            "--preflight-target-only",
            "--expected-target-row-count",
            str(expected_target_crop_rows),
            "--json",
        ),
        resources=cpu,
        upstream=(hybrid_key,),
    )
    jobs.append(keypoint_preflight_job)
    keypoint_tasks: list[LsfExecutionTask] = []
    mask_tasks: list[LsfExecutionTask] = []
    for clip in clips:
        clip_id = str(clip["clip_id"])
        row_start = int(clip["crop_row_start"])
        row_stop = int(clip["crop_row_stop"])
        keypoint_command = [
            "scripts/py",
            "-m",
            "fisheye.utils.run_keypoints_with_registry_model",
            "--recording-dir",
            str(target.recording_dir),
            "--output",
            str(target.analysis_zarr),
            "--registry",
            str(registry_path),
            "--set-id",
            pose_binding.set_id,
            "--model-run-id",
            pose_binding.run_id,
            "--require-unique",
            "--run-name",
            str(clip["keypoint_shard_run"]),
            "--output-parent",
            "keypoint_shard_runs",
            "--coordinate-contract-mode",
            "legacy_noncanonical",
            "--crop-run",
            hybrid_crop_run,
            "--source-crop-row-start",
            str(row_start),
            "--source-crop-row-stop",
            str(row_stop),
            "--roi-cache-policy",
            "never",
            "--pose-schema",
            "traditional_v2",
            "--batch-size",
            "256",
            "--device",
            "0",
            "--keypoint-roi-shard-rows",
            "131072",
            "--keypoint-frame-shard-rows",
            "131072",
            "--progress-jsonl",
            str(run_root / "progress" / f"keypoints_{target_safe}_{clip_id}.jsonl"),
        ]
        keypoint_tasks.append(
            _execution_task(
                run_root=run_root,
                task_key=f"keypoints:{target_safe}:{clip_id}",
                stage="keypoints",
                command=keypoint_command,
                expected_outputs=(
                    target.analysis_zarr
                    / "keypoint_shard_runs"
                    / str(clip["keypoint_shard_run"])
                    / "zarr.json",
                ),
                array_indexed=True,
            )
        )

        mask_worker_receipt = (
            run_root / "receipts" / f"subject_masks_{target_safe}_{clip_id}.json"
        )
        mask_command = [
            "scripts/py",
            "-m",
            "fisheye.cluster.subject_masks.staged_inference",
            "--direct-crop-provider",
            "--worker-receipt-json",
            str(mask_worker_receipt),
            str(target.analysis_zarr),
            "--resolve-model-from-registry",
            "--registry",
            str(registry_path),
            "--model-set-id",
            subject_binding.set_id,
            "--model-run-id",
            subject_binding.run_id,
            "--model-coverage-class",
            subject_mask_coverage_class,
            "--model-component-coverage-key",
            subject_mask_component_coverage_key,
            "--model-label-schema-id",
            subject_mask_label_schema_id,
            "--model-require-unique",
            "--run-name",
            str(clip["subject_mask_shard_run"]),
            "--output-parent",
            "subject_mask_shard_runs",
            "--crop-run",
            hybrid_crop_run,
            "--source-crop-row-start",
            str(row_start),
            "--source-crop-row-stop",
            str(row_stop),
            "--source-collection-id",
            target.finalized_refined_detect_run,
            "--source-collection-path",
            source_run_path,
            "--source-clip-id",
            clip_id,
            "--source-clip-index",
            str(clip["clip_index"]),
            "--source-work-unit-id",
            str(clip["work_unit_id"]),
            "--roi-cache-policy",
            "never",
            "--batch-size",
            "128",
            "--device",
            "0",
            "--mask-probs-dtype",
            "uint8",
            "--mask-probs-chunk-rois",
            "32",
            "--mask-probs-shard-rois",
            "2048",
            "--no-write-masks-roi",
            "--async-output",
            "--output-queue-size",
            "2",
            "--no-progress",
            "--defer-registry-status",
        ]
        mask_tasks.append(
            _execution_task(
                run_root=run_root,
                task_key=f"subject_masks:{target_safe}:{clip_id}",
                stage="subject_mask_inference",
                command=mask_command,
                expected_outputs=(
                    target.analysis_zarr
                    / "subject_mask_shard_runs"
                    / str(clip["subject_mask_shard_run"])
                    / "zarr.json",
                    mask_worker_receipt,
                ),
                array_indexed=True,
            )
        )

    keypoint_array_job = _task_group_job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=keypoint_array_key,
        stage="keypoints",
        tasks=keypoint_tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=gpu_array_concurrency,
        resources=LsfResources(
            queue=keypoint_gpu_queue,
            ncores=8,
            mem_gb=48,
            gpus=1,
            walltime="4:00",
        ),
        upstream=(keypoint_preflight_key,),
    )
    mask_array_job = _task_group_job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=subject_mask_array_key,
        stage="subject_mask_inference",
        tasks=mask_tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=gpu_array_concurrency,
        resources=LsfResources(
            queue=subject_mask_gpu_queue,
            ncores=8,
            mem_gb=48,
            gpus=1,
            walltime="4:00",
        ),
        upstream=(hybrid_key,),
    )
    jobs.extend((keypoint_array_job, mask_array_job))

    keypoint_finalize_key = f"keypoint_finalize:{target_safe}"
    finalize_keypoints = [
        "scripts/py",
        "-m",
        "fisheye.utils.finalize_keypoint_shards",
        str(target.analysis_zarr),
        "--target-crop-run",
        hybrid_crop_run,
        "--output-run",
        keypoint_run,
        "--json",
    ]
    for clip in clips:
        finalize_keypoints.extend(["--shard-run", str(clip["keypoint_shard_run"])])
    keypoint_finalize_job = _job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=keypoint_finalize_key,
        stage="keypoint_finalize",
        command=finalize_keypoints,
        resources=final_cpu,
        upstream=(keypoint_array_key,),
        expected_outputs=(
            target.analysis_zarr / "keypoints_runs" / keypoint_run / "zarr.json",
        ),
    )
    keypoint_refine_key = f"keypoint_refine:{target_safe}"
    keypoint_refine_job = _job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=keypoint_refine_key,
        stage="keypoint_refine",
        command=(
            "scripts/py",
            "-m",
            "fisheye.refinement.refine_keypoints",
            str(target.analysis_zarr),
            "--keypoint-run",
            keypoint_run,
            "--run-name",
            refined_keypoint_run,
            "--chunk-size",
            "2048",
            "--scheduler",
            "threads",
            "--num-workers",
            "4",
            "--no-post-audit",
        ),
        resources=cpu,
        upstream=(keypoint_finalize_key,),
        expected_outputs=(
            target.analysis_zarr
            / "refined_keypoints_runs"
            / refined_keypoint_run
            / "zarr.json",
        ),
    )
    jobs.extend((keypoint_finalize_job, keypoint_refine_job))
    raw_keypoints_artifact = f"raw_keypoints:{target_safe}"
    refined_keypoints_artifact = f"refined_keypoints:{target_safe}"
    raw_masks_artifact = f"raw_subject_masks:{target_safe}"
    fragments.extend(
        (
            LsfWorkflowFragment(
                fragment_id=f"keypoints:{target_safe}",
                jobs=(
                    keypoint_preflight_job,
                    keypoint_array_job,
                    keypoint_finalize_job,
                    keypoint_refine_job,
                ),
                requires=(hybrid_artifact,),
                provides=(raw_keypoints_artifact, refined_keypoints_artifact),
                metadata={
                    "module": "keypoints",
                    "target_id": target.target_id,
                    "work_partition": "clip_crop_row_intervals",
                    "crop_run": hybrid_crop_run,
                    "finalization_mapping_mode": "direct_same_crop_row_ids",
                    "expected_target_crop_rows": expected_target_crop_rows,
                },
            ),
            LsfWorkflowFragment(
                fragment_id=f"subject_mask_inference:{target_safe}",
                jobs=(mask_array_job,),
                requires=(hybrid_artifact,),
                provides=(raw_masks_artifact,),
                metadata={
                    "module": "subject_mask_inference",
                    "target_id": target.target_id,
                    "work_partition": "clip_crop_row_intervals",
                    "crop_run": hybrid_crop_run,
                },
            ),
        )
    )

    mask_grid_key: str | None = None
    if encoded_mask_packages:
        mask_grid_key = f"mask_grid:{target_safe}"
        jobs.append(
            _job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=mask_grid_key,
                stage="subject_mask_global_chunk_grid",
                command=(
                    "scripts/py",
                    "-m",
                    "fisheye.utils.prepare_refined_subject_mask_chunk_grid",
                    "--zarr",
                    str(target.analysis_zarr),
                    "--crop-run",
                    hybrid_crop_run,
                    "--output-manifest",
                    str(global_mask_grid_manifest),
                    "--mask-label",
                    "subject_body",
                    "--mask-label",
                    "eye_left",
                    "--mask-label",
                    "eye_right",
                    "--mask-label",
                    "swim_bladder",
                    "--mask-height",
                    "512",
                    "--mask-width",
                    "512",
                    "--dense-mask-row-chunk",
                    "128",
                    "--json",
                ),
                resources=cpu,
                upstream=(keypoint_finalize_key,),
                expected_outputs=(global_mask_grid_manifest,),
            )
        )

    package_array_key = f"mask_package_array:{target_safe}"
    package_tasks: list[LsfExecutionTask] = []
    for clip in clips:
        clip_id = str(clip["clip_id"])
        package_command = [
            "scripts/py",
            "-m",
            "fisheye.utils.finalize_subject_mask_clip_package",
            "--source-zarr",
            str(target.analysis_zarr),
            "--subject-shard-run",
            str(clip["subject_mask_shard_run"]),
            "--target-crop-run",
            hybrid_crop_run,
            "--refined-run",
            str(clip["refined_mask_package_run"]),
            "--package-path",
            str(clip["package_path"]),
            "--component",
            "subject_body",
            "--component",
            "eyes_union",
            "--component",
            "swim_bladder",
            "--chunk-size",
            "256",
            "--metric-level",
            "cheap",
            "--mask-storage",
            "dense_and_bitpacked",
            "--dense-mask-row-chunk",
            "128",
            "--execution-backend",
            "process_shards",
            "--num-workers",
            "8",
            "--postcompute-backend",
            "process_shards",
            "--postcompute-num-workers",
            "8",
            "--postcompute-chunk-size",
            "256",
            "--assignment-keypoint-group",
            "refined_keypoints_runs",
            "--assignment-keypoints-run",
            refined_keypoint_run,
            "--no-write-component-contours",
            "--require-production-proof",
            "--json",
        ]
        if (
            subject_mask_publication_profile
            == SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED
        ):
            package_command.extend(
                [
                    "--publication-evidence-producer-commit",
                    repo_commit,
                    "--work-unit-id",
                    str(clip["work_unit_id"]),
                    "--work-unit-index",
                    str(clip["work_unit_index"]),
                    "--source-clip-id",
                    clip_id,
                    "--source-clip-index",
                    str(clip["clip_index"]),
                    "--global-frame-start",
                    str(clip["frame_start"]),
                    "--global-frame-stop",
                    str(clip["frame_stop"]),
                    "--quality-compute-workers",
                    "4",
                ]
            )
        if encoded_mask_packages:
            package_command.extend(
                [
                    "--global-mask-grid-manifest",
                    str(global_mask_grid_manifest),
                    "--encoded-mask-copy-workers",
                    "8",
                ]
            )
        package_tasks.append(
            _execution_task(
                run_root=run_root,
                task_key=f"mask_package:{target_safe}:{clip_id}",
                stage="subject_mask_refine_package",
                command=package_command,
                expected_outputs=(Path(str(clip["package_path"])),),
                array_indexed=True,
            )
        )
    package_upstream = [subject_mask_array_key, keypoint_refine_key]
    if mask_grid_key is not None:
        package_upstream.append(mask_grid_key)
    package_array_job = _task_group_job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=package_array_key,
        stage="subject_mask_refine_package",
        tasks=package_tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=mask_package_array_concurrency,
        resources=final_cpu,
        upstream=tuple(package_upstream),
    )
    jobs.append(package_array_job)

    mask_import_key = f"mask_import:{target_safe}"
    mask_import_job: LsfJob | None = None
    if subject_mask_publication_profile == SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK:
        mask_import_command = [
            "scripts/py",
            "-m",
            "fisheye.utils.import_refined_subject_mask_clip_packages",
            "--zarr",
            str(target.analysis_zarr),
            "--output-run",
            refined_subject_mask_draft_run,
            "--expected-target-crop-run",
            hybrid_crop_run,
            "--array-copy-workers",
            "8",
            "--encoded-copy-workers",
            "32",
            "--require-production-proof",
            "--json",
        ]
        for clip in clips:
            mask_import_command.extend(["--package", str(clip["package_path"])])
        mask_import_job = _job(
            workflow_id=workflow_id,
            repo=repo,
            run_root=run_root,
            job_key=mask_import_key,
            stage="subject_mask_collection_import",
            command=mask_import_command,
            resources=import_cpu,
            upstream=(package_array_key,),
            expected_outputs=(
                target.analysis_zarr
                / "refined_subject_masks_runs"
                / refined_subject_mask_draft_run
                / "zarr.json",
            ),
        )
        jobs.append(mask_import_job)
    else:
        mask_import_key = package_array_key

    mask_publish_key = f"mask_publish:{target_safe}"
    receipt_composed = (
        subject_mask_publication_profile == SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED
    )
    mask_publish_output = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "palette_subject_mask_bundle_outputs"
    )
    mask_quality_scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "palette_subject_mask_quality"
    )
    mask_publish_command = [
        "scripts/py",
        "-m",
        (
            "fisheye.cluster.subject_masks.publish_receipt_composed_bundle"
            if receipt_composed
            else "fisheye.cluster.subject_masks.publish_recording_bundle"
        ),
        "--analysis-zarr",
        str(target.analysis_zarr),
        "--crop-run",
        hybrid_crop_run,
        "--raw-run",
        subject_mask_run,
        "--refined-run",
        refined_subject_mask_run,
        "--quality-run",
        subject_mask_quality_run,
        "--cache-run",
        subject_mask_cache_run,
        "--bundle-id",
        subject_mask_bundle_id,
        "--local-output-root",
        mask_publish_output,
        "--quality-scratch-root",
        mask_quality_scratch,
        "--json",
    ]
    if receipt_composed:
        mask_publish_command.extend(["--producer-commit", repo_commit])
        for clip in clips:
            mask_publish_command.extend(
                ["--refined-package", str(clip["package_path"])]
            )
    else:
        mask_publish_command.extend(
            [
                "--draft-zarr",
                str(target.analysis_zarr),
                "--raw-draft-parent",
                "subject_mask_shard_runs",
                "--refined-draft-run",
                refined_subject_mask_draft_run,
            ]
        )
    for clip in clips:
        mask_publish_command.extend(
            ["--raw-draft-run", str(clip["subject_mask_shard_run"])]
        )
    mask_publish_job = _job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=mask_publish_key,
        stage="subject_mask_collection_publication",
        command=mask_publish_command,
        resources=import_cpu,
        upstream=(mask_import_key,),
        cleanup_paths=(mask_publish_output, mask_quality_scratch),
        expected_outputs=(
            target.analysis_zarr
            / "subject_mask_bundle_runs"
            / subject_mask_bundle_id
            / "zarr.json",
        ),
    )
    jobs.append(mask_publish_job)
    mask_fragment_jobs: list[LsfJob] = [package_array_job]
    if mask_import_job is not None:
        mask_fragment_jobs.append(mask_import_job)
    mask_fragment_jobs.append(mask_publish_job)
    refined_masks_artifact = f"refined_subject_masks:{target_safe}"
    fragments.append(
        LsfWorkflowFragment(
            fragment_id=f"subject_mask_refinement:{target_safe}",
            jobs=tuple(mask_fragment_jobs),
            requires=(raw_masks_artifact, refined_keypoints_artifact),
            provides=(refined_masks_artifact,),
            metadata={
                "module": "subject_mask_refinement",
                "target_id": target.target_id,
                "crop_run": hybrid_crop_run,
                "selector_activation": False,
            },
        )
    )

    validation_key = f"validate:{target_safe}"
    validation_report = run_root / "validation" / f"{target_safe}.json"
    validation_job = _job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=validation_key,
        stage="validation",
        command=(
            "scripts/py",
            "-m",
            "fisheye.cluster.clipped_inference_validate",
            "--plan",
            str(run_root / "plan.json"),
            "--target-id",
            target.target_id,
            "--output-json",
            str(validation_report),
        ),
        resources=final_cpu,
        upstream=(mask_publish_key,),
        expected_outputs=(validation_report,),
    )
    jobs.append(validation_job)
    validated_artifact = f"validated_analysis:{target_safe}"
    fragments.append(
        LsfWorkflowFragment(
            fragment_id=f"analysis_validation:{target_safe}",
            jobs=(validation_job,),
            requires=(refined_keypoints_artifact, refined_masks_artifact),
            provides=(validated_artifact,),
            metadata={
                "module": "analysis_validation",
                "target_id": target.target_id,
                "source_refined_detect_run": target.finalized_refined_detect_run,
                "crop_run": hybrid_crop_run,
                "selector_activation": False,
            },
        )
    )
    return _DownstreamTargetPipeline(
        jobs=tuple(jobs),
        fragments=tuple(fragments),
        terminal_job_key=validation_key,
        terminal_artifact_key=validated_artifact,
    )


def _artifact_backed_detection_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Make compatibility refinement consume the native artifact rows once."""

    resolved = copy.deepcopy(dict(plan))
    units = resolved.get("work_units")
    if not isinstance(units, list) or not units:
        raise ValueError("Detection plan requires non-empty work units.")
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Detection plan work unit is not mutable JSON data.")
        paths = unit.get("zarr_paths")
        commands = unit.get("commands")
        if not isinstance(paths, dict) or not isinstance(commands, dict):
            raise ValueError("Detection plan work unit lacks paths or commands.")
        legacy_family = str(paths.get("detect_family_path") or "")
        legacy_group = str(paths.get("detect_target_group_path") or "")
        artifact_family = str(paths.get("detection_artifact_family_path") or "")
        artifact_group = str(paths.get("detection_artifact_target_group_path") or "")
        if not all((legacy_family, legacy_group, artifact_family, artifact_group)):
            raise ValueError("Detection plan lacks artifact/compatibility paths.")
        paths["detect_family_path"] = artifact_family
        paths["detect_target_group_path"] = artifact_group
        for name in ("validate_detect", "detect_quality", "refine_detect"):
            rendered = commands.get(name)
            if isinstance(rendered, str):
                commands[name] = rendered.replace(legacy_group, artifact_group).replace(
                    legacy_family,
                    artifact_family,
                )
    resolved["raw_detection_storage_profile"] = "artifact_first_native_canonical_v1"
    resolved["compatibility_refinement_source"] = "detection_artifact_runs"
    return resolved


def _order_detection_work_units_by_recording_frame(
    *,
    target_id: str,
    work_units: Sequence[Mapping[str, Any]],
    frame_intervals: Mapping[tuple[int, str], tuple[int, int]],
) -> list[Mapping[str, Any]]:
    """Require exact indexed-unit coverage, then order units on the native timeline."""

    if not work_units:
        raise ValueError(f"Target {target_id!r} detection plan has no work units.")
    planned_keys = [
        (int(unit["clip_index"]), str(unit["clip_id"])) for unit in work_units
    ]
    if len(planned_keys) != len(set(planned_keys)):
        raise ValueError(
            f"Target {target_id!r} detection plan has duplicate "
            "(clip_index, clip_id) work units."
        )
    planned_key_set = set(planned_keys)
    frame_interval_key_set = set(frame_intervals)
    if planned_key_set != frame_interval_key_set:
        missing = sorted(frame_interval_key_set - planned_key_set)
        unexpected = sorted(planned_key_set - frame_interval_key_set)
        raise ValueError(
            f"Target {target_id!r} detection plan must exactly cover the "
            "recording frame-index work units; "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    return sorted(
        work_units,
        key=lambda unit: frame_intervals[
            (int(unit["clip_index"]), str(unit["clip_id"]))
        ][0],
    )


def _repo_commit(repo: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(repo.expanduser().resolve()), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    commit = commit.lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Palette repo did not resolve one full commit SHA.")
    return commit


def _validate_existing_detection_for_resume(
    *,
    target: CampaignTarget,
    workflow_id: str,
    clip: Mapping[str, Any],
    binding: ModelBinding,
) -> dict[str, Any]:
    """Fail closed unless an imported detection exactly matches the planned work unit."""

    group_path = str(clip["detect_group_path"])
    validation = validate_imported_run_group(
        zarr_path=target.analysis_zarr,
        target_group_path=group_path,
        validate_source_tarball=False,
    )
    if validation.get("status") not in {"ok", "pass"}:
        raise ValueError(
            f"Existing detection failed import validation for {clip['clip_id']}: "
            + json.dumps(validation, sort_keys=True, default=str)
        )

    metadata_path = target.analysis_zarr / group_path / "zarr.json"
    payload = _read_strict_json(metadata_path)
    attrs = payload.get("attributes") if isinstance(payload, Mapping) else None
    if not isinstance(attrs, Mapping):
        raise ValueError(
            f"Existing detection has no attributes object: {metadata_path}"
        )
    if attrs.get("palette_run_completion_status") != "complete":
        raise ValueError(f"Existing detection is not complete: {metadata_path}")
    provenance = attrs.get("run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Existing detection has no run_provenance: {metadata_path}")
    params = provenance.get("params")
    input_run_ids = provenance.get("input_run_ids")
    clip_context = params.get("clip_context") if isinstance(params, Mapping) else None
    if not isinstance(params, Mapping) or not isinstance(input_run_ids, Mapping):
        raise ValueError(
            f"Existing detection has incomplete run provenance: {metadata_path}"
        )
    if not isinstance(clip_context, Mapping):
        raise ValueError(f"Existing detection has no clip context: {metadata_path}")

    expected = {
        "command": "fisheye.utils.run_detection_artifact",
        "params.run_name": str(clip["detect_run"]),
        "params.video_path": str(clip["video_path"]),
        "params.target_zarr": str(target.analysis_zarr),
        "params.model_path": str(binding.path),
        "params.model_sha256": binding.sha256,
        "params.model_registry_set_id": binding.set_id,
        "params.model_registry_run_id": binding.run_id,
        "params.clip_context.workflow_id": workflow_id,
        "params.clip_context.recording_id": target.recording_id,
        "params.clip_context.clip_id": str(clip["clip_id"]),
        "params.clip_context.clip_index": int(clip["clip_index"]),
        "params.clip_context.camera_serial": str(clip["camera_serial"]),
        "input_run_ids.model_registry_set_id": binding.set_id,
        "input_run_ids.model_registry_run_id": binding.run_id,
    }
    observed = {
        "command": provenance.get("command"),
        "params.run_name": params.get("run_name"),
        "params.video_path": params.get("video_path"),
        "params.target_zarr": params.get("target_zarr"),
        "params.model_path": params.get("model_path"),
        "params.model_sha256": params.get("model_sha256"),
        "params.model_registry_set_id": params.get("model_registry_set_id"),
        "params.model_registry_run_id": params.get("model_registry_run_id"),
        "params.clip_context.workflow_id": clip_context.get("workflow_id"),
        "params.clip_context.recording_id": clip_context.get("recording_id"),
        "params.clip_context.clip_id": clip_context.get("clip_id"),
        "params.clip_context.clip_index": clip_context.get("clip_index"),
        "params.clip_context.camera_serial": clip_context.get("camera_serial"),
        "input_run_ids.model_registry_set_id": input_run_ids.get(
            "model_registry_set_id"
        ),
        "input_run_ids.model_registry_run_id": input_run_ids.get(
            "model_registry_run_id"
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    artifacts = provenance.get("input_artifacts")
    matching_artifacts = (
        [
            artifact
            for artifact in artifacts
            if isinstance(artifact, Mapping)
            and artifact.get("role") == "detect_model"
            and artifact.get("path") == str(binding.path)
            and artifact.get("sha256") == binding.sha256
        ]
        if isinstance(artifacts, list)
        else []
    )
    if not matching_artifacts:
        mismatches["input_artifacts.detect_model"] = {
            "expected": {"path": str(binding.path), "sha256": binding.sha256},
            "observed": artifacts,
        }
    if mismatches:
        raise ValueError(
            f"Existing detection provenance mismatch for {clip['clip_id']}: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )

    return {
        "status": "ok",
        "clip_id": str(clip["clip_id"]),
        "target_group_path": group_path,
        "receipt_path": validation.get("receipt_path"),
        "validator_status": validation.get("status"),
        "completion_status": "complete",
        "model_sha256": binding.sha256,
    }


def build_plan(
    *,
    targets: Sequence[CampaignTarget],
    run_label: str,
    repo: Path,
    registry_path: Path,
    run_root: Path,
    detection_set_id: str | None = None,
    detection_run_id: str | None = None,
    pose_set_id: str | None = None,
    pose_run_id: str | None = None,
    subject_mask_set_id: str | None = None,
    subject_mask_run_id: str | None = None,
    workflow_scope: str = WORKFLOW_SCOPE_FULL,
    subject_mask_coverage_class: str = "dense_all_components",
    subject_mask_component_coverage_key: str = "body+eyes+swim_bladder",
    subject_mask_label_schema_id: str = "subject_v1_union",
    cache_root: Path = DEFAULT_CACHE_ROOT,
    package_root: Path = DEFAULT_PACKAGE_ROOT,
    cache_bundle_size: int = 8,
    max_active_targets: int = 3,
    cleanup_nrs_after_success: bool = True,
    resume_existing_detections: bool = False,
    encoded_mask_packages: bool = False,
    subject_mask_publication_profile: str = (SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED),
    detect_array_concurrency: int = 8,
    gpu_array_concurrency: int = 4,
    cache_array_concurrency: int = 2,
    mask_package_array_concurrency: int = 4,
    detect_refine_bundle_concurrency: int = 4,
    keypoint_gpu_queue: str = "gpu_t4",
    subject_mask_gpu_queue: str = "gpu_l4",
    registered_gate_requirement: str = "off",
    registered_gate_run: str | None = None,
    selection_policy_id: str = "manual_review_only_v1",
) -> ClippedInferencePlan:
    if not targets:
        raise ValueError("At least one target is required.")
    scope = str(workflow_scope).strip()
    if scope not in WORKFLOW_SCOPES:
        raise ValueError(f"workflow_scope must be one of {WORKFLOW_SCOPES!r}.")
    downstream_model_ids = {
        "pose_set_id": pose_set_id,
        "pose_run_id": pose_run_id,
        "subject_mask_set_id": subject_mask_set_id,
        "subject_mask_run_id": subject_mask_run_id,
    }
    missing_downstream_ids = sorted(
        name
        for name, value in downstream_model_ids.items()
        if not str(value or "").strip()
    )
    if (
        scope in {WORKFLOW_SCOPE_FULL, WORKFLOW_SCOPE_DOWNSTREAM}
        and missing_downstream_ids
    ):
        raise ValueError(
            "Full/downstream workflow scope requires exact downstream model identifiers: "
            + ", ".join(missing_downstream_ids)
        )
    if (
        scope in {WORKFLOW_SCOPE_FULL, WORKFLOW_SCOPE_DOWNSTREAM}
        and subject_mask_publication_profile not in SUBJECT_MASK_PUBLICATION_PROFILES
    ):
        raise ValueError(
            "Unsupported subject-mask publication profile: "
            f"{subject_mask_publication_profile!r}."
        )
    if scope != WORKFLOW_SCOPE_DOWNSTREAM and (
        not str(detection_set_id or "").strip()
        or not str(detection_run_id or "").strip()
    ):
        raise ValueError(
            "Detection/full workflow scope requires exact detection model identifiers."
        )
    if scope == WORKFLOW_SCOPE_DOWNSTREAM:
        missing_runs = [
            target.target_id
            for target in targets
            if not target.finalized_refined_detect_run
        ]
        if missing_runs:
            raise ValueError(
                "Downstream workflow scope requires finalized_refined_detect_run "
                "for every target: " + ", ".join(missing_runs)
            )
        if resume_existing_detections:
            raise ValueError(
                "Downstream scope consumes an exact finalized detection run; "
                "--resume-existing-detections is not applicable."
            )
    gate_requirement = str(registered_gate_requirement).strip()
    if gate_requirement not in REGISTERED_GATE_REQUIREMENTS:
        raise ValueError(
            "registered_gate_requirement must be off, if_available, or required."
        )
    gate_run = str(registered_gate_run or "").strip() or None
    policy_id = str(selection_policy_id).strip()
    if policy_id not in {"manual_review_only_v1", "corroborated_acquisition_v1"}:
        raise ValueError("Unsupported registered geometry selection policy id.")
    concurrency_values = {
        "cache_bundle_size": cache_bundle_size,
        "max_active_targets": max_active_targets,
        "detect_array_concurrency": detect_array_concurrency,
        "gpu_array_concurrency": gpu_array_concurrency,
        "cache_array_concurrency": cache_array_concurrency,
        "mask_package_array_concurrency": mask_package_array_concurrency,
        "detect_refine_bundle_concurrency": detect_refine_bundle_concurrency,
    }
    if any(int(value) <= 0 for value in concurrency_values.values()):
        raise ValueError(
            "Workflow bundle sizes and concurrency limits must be positive: "
            f"{concurrency_values}"
        )
    for queue_name, queue_value in (
        ("keypoint_gpu_queue", keypoint_gpu_queue),
        ("subject_mask_gpu_queue", subject_mask_gpu_queue),
    ):
        if queue_value not in {"gpu_l4", "gpu_t4"}:
            raise ValueError(f"{queue_name} must be gpu_l4 or gpu_t4.")
    label = safe_component(run_label, default="clipped_inference", max_length=72)
    workflow_id = label
    repo = repo.expanduser().resolve()
    registry_path = registry_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    config_path = repo / "configs" / "fisheye" / "yolo_detect_config.yaml"
    refine_config = repo / "configs" / "fisheye" / "default.yaml"
    for required in (repo / "scripts" / "py", config_path, refine_config):
        if not required.exists():
            raise FileNotFoundError(required)
    repo_commit = _repo_commit(repo)
    modern_registered_pipeline = gate_requirement != "off"
    canonical_recording_pipeline = (
        scope == WORKFLOW_SCOPE_DETECTION or modern_registered_pipeline
    )

    detect_bindings: list[ModelBinding] = []
    pose_bindings: list[ModelBinding] = []
    for target in targets:
        validate_registered_analysis_zarr(
            registry_path=registry_path,
            recording_id=target.recording_id,
            analysis_zarr=target.analysis_zarr,
        )
        if scope != WORKFLOW_SCOPE_DOWNSTREAM:
            detect_bindings.append(
                _resolve_ranked_binding(
                    registry_path=registry_path,
                    target=target,
                    task="detect",
                    set_id=str(detection_set_id),
                    run_id=str(detection_run_id),
                )
            )
        if scope in {WORKFLOW_SCOPE_FULL, WORKFLOW_SCOPE_DOWNSTREAM}:
            pose = resolve_pose_model_binding(
                registry_path=registry_path,
                recording_id=target.recording_id,
                recording_dir=target.recording_dir,
                set_id=str(pose_set_id),
                run_id=str(pose_run_id),
            )
            pose_binding = ModelBinding(
                task="pose",
                set_id=pose.set_id,
                run_id=pose.run_id,
                path=pose.model_path,
                sha256=pose.model_sha256,
            )
            _verify_binding(pose_binding)
            pose_bindings.append(pose_binding)
    detection_binding = (
        _assert_same_binding("detection", detect_bindings) if detect_bindings else None
    )
    pose_binding = (
        _assert_same_binding("pose", pose_bindings)
        if scope in {WORKFLOW_SCOPE_FULL, WORKFLOW_SCOPE_DOWNSTREAM}
        else None
    )
    subject_binding = (
        _resolve_subject_binding(
            registry_path=registry_path,
            set_id=str(subject_mask_set_id),
            run_id=str(subject_mask_run_id),
            coverage_class=subject_mask_coverage_class,
            component_coverage_key=subject_mask_component_coverage_key,
            label_schema_id=subject_mask_label_schema_id,
        )
        if scope in {WORKFLOW_SCOPE_FULL, WORKFLOW_SCOPE_DOWNSTREAM}
        else None
    )

    gpu = LsfResources(queue="gpu_l4", ncores=8, mem_gb=48, gpus=1, walltime="4:00")
    cpu = LsfResources(queue="short", ncores=4, mem_gb=32, walltime="1:00")
    final_cpu = LsfResources(queue="short", ncores=8, mem_gb=32, walltime="1:00")
    import_cpu = LsfResources(queue="local", ncores=8, mem_gb=32, walltime="3:00")
    cache_gpu = LsfResources(
        queue="gpu_l4",
        ncores=8,
        mem_gb=64,
        gpus=0,
        walltime="4:00",
        extra_lsf_args=("-gpu", "num=1:mode=shared:j_exclusive=no"),
    )

    jobs: list[LsfJob] = []
    fragments: list[LsfWorkflowFragment] = []
    detection_scope_outputs: list[dict[str, Any]] = []
    target_payloads: list[dict[str, Any]] = []
    target_terminal_keys: list[str] = []
    target_terminal_artifacts: list[str] = []

    for target_index, target in enumerate(targets):
        target_safe = safe_component(
            target.target_id, default=f"target_{target_index}", max_length=56
        )
        target_label = safe_component(
            f"{label}_{target_safe}", default=target_safe, max_length=90
        )
        authority = load_native_archive_authority(target)
        validate_recording_frame_index(
            target.recording_dir / "recording_frame_index.parquet",
            n_frames=authority.n_frames,
        )
        frame_intervals = recording_frame_work_unit_intervals(
            target.recording_dir / "recording_frame_index.parquet",
            n_frames=authority.n_frames,
        )
        collection_id = f"refined_detect_collection_{target_label}"
        detect_quality_source_run = f"detect_quality_source_{target_label}"
        detect_quality_run = f"detect_quality_{target_label}"
        cache_dir = cache_root.expanduser().resolve() / label / target_safe
        package_dir = package_root.expanduser().resolve() / label / target_safe
        global_mask_grid_manifest = package_dir / "global_mask_chunk_grid.json"
        detection_plan_path = run_root / "targets" / target_safe / "detection_plan.json"
        if scope == WORKFLOW_SCOPE_DOWNSTREAM:
            work_units = _clipped_layout_work_units(
                target, camera_serial=authority.camera_serial
            )
            detection_plan = {
                "schema": "palette.clipped_downstream_layout_plan.v1",
                "workflow_id": target_label,
                "recording_id": target.recording_id,
                "analysis_zarr": str(target.analysis_zarr),
                "external_refined_detect_run": target.finalized_refined_detect_run,
                "work_units": work_units,
            }
        else:
            assert detection_binding is not None
            detection_plan = _artifact_backed_detection_plan(
                build_detection_plan(
                    target.recording_dir,
                    analysis_zarr=target.analysis_zarr,
                    model=detection_binding.path,
                    config=config_path,
                    workflow_id=target_label,
                    output_dir=run_root
                    / "targets"
                    / target_safe
                    / "detection_artifacts",
                )
            )
            work_units = list(detection_plan["work_units"])
        if {str(unit["camera_serial"]) for unit in work_units} != {
            authority.camera_serial
        }:
            raise ValueError(
                "Detection plan camera differs from acquisition authority."
            )

        hybrid_crop_run = f"crop_hybrid_{target_label}"
        merged_proxy = (
            hybrid_crop_run
            if scope == WORKFLOW_SCOPE_DOWNSTREAM
            else f"crop_proxy_{target_label}_collection"
        )
        native_canonical_run = f"detect_native_{target_label}"
        strict_storage_bundle = (
            target.recording_dir.parent
            / ".palette_benchmarks"
            / "clipped_storage_candidates"
            / label
            / target_safe
        )
        keypoint_run = f"keypoints_registry_{target_label}"
        refined_keypoint_run = f"refined_keypoints_{target_label}"
        subject_mask_run = f"subject_masks_{target_label}"
        refined_subject_mask_run = f"refined_subject_masks_{target_label}"
        refined_subject_mask_draft_run = f"{refined_subject_mask_run}__worker_draft"
        subject_mask_quality_run = f"subject_mask_quality_{target_label}"
        subject_mask_cache_run = f"subject_mask_cache_{target_label}"
        subject_mask_bundle_id = f"subject_mask_bundle_{target_label}"
        canonical_refined_run = (
            f"refined_detect_native_{target_label}"
            if canonical_recording_pipeline
            else None
        )
        materialize_registered_gate = (
            gate_requirement == "required" and gate_run is None
        )
        geometry_selection_run = (
            _active_arena_geometry_selection(target.analysis_zarr)
            if materialize_registered_gate
            else None
        )
        target_gate_run = (
            f"registered_detection_gate_{target_label}"
            if materialize_registered_gate
            else gate_run
        )
        planned_gate_group_path = (
            f"analysis/detection_gate_runs/{target_gate_run}"
            if materialize_registered_gate
            else None
        )
        clips: list[dict[str, Any]] = []
        ordered_work_units = _order_detection_work_units_by_recording_frame(
            target_id=target.target_id,
            work_units=work_units,
            frame_intervals=frame_intervals,
        )
        for work_unit_index, unit in enumerate(ordered_work_units):
            clip = str(unit["clip_id"])
            clip_index = int(unit["clip_index"])
            try:
                frame_start, frame_stop = frame_intervals[(clip_index, clip)]
            except KeyError as exc:
                raise ValueError(
                    f"Recording frame index has no interval for clip {clip!r}."
                ) from exc
            manifest = (
                cache_dir / f"roi_cache_{target_label}__{clip}.flat_roi_cache.json"
            )
            alias = manifest.with_name(
                f"{manifest.stem}__crop_proxy_{target_label}_{clip}.alias.json"
            )
            clips.append(
                {
                    "clip_id": clip,
                    "clip_index": int(unit["clip_index"]),
                    "work_unit_index": int(work_unit_index),
                    "frame_start": int(frame_start),
                    "frame_stop": int(frame_stop),
                    "camera_serial": str(unit["camera_serial"]),
                    "work_unit_id": str(unit["work_unit_id"]),
                    "video_path": str(unit["source"]["video_path"]),
                    "detect_run": (
                        str(unit["run_names"]["detect"])
                        if "run_names" in unit
                        else None
                    ),
                    "detect_group_path": (
                        str(unit["zarr_paths"]["detect_target_group_path"])
                        if "zarr_paths" in unit
                        else None
                    ),
                    "quality_run": (
                        str(unit["run_names"]["detect_quality"])
                        if "run_names" in unit
                        else None
                    ),
                    "refined_detect_run": (
                        str(unit["run_names"]["refined_detect"])
                        if "run_names" in unit
                        else target.finalized_refined_detect_run
                    ),
                    "refined_detect_group_path": (
                        str(unit["zarr_paths"]["refined_group_path"])
                        if "zarr_paths" in unit
                        else f"refined_detect_runs/{target.finalized_refined_detect_run}"
                    ),
                    "cache_manifest": str(manifest),
                    "cache_row_index": str(
                        manifest.with_suffix("").with_suffix(
                            ".flat_roi_cache.rows.parquet"
                        )
                    ),
                    "proxy_crop_run": f"crop_proxy_{target_label}_{clip}",
                    "alias_manifest": str(alias),
                    "keypoint_shard_run": f"keypoint_shard_{target_label}_{clip}",
                    "subject_mask_shard_run": f"subject_mask_shard_{target_label}_{clip}",
                    "refined_mask_package_run": f"refined_subject_masks_{target_label}_{clip}",
                    "package_path": str(package_dir / f"{clip}.tar.gz"),
                }
            )
            if scope == WORKFLOW_SCOPE_DOWNSTREAM:
                for legacy_cache_key in (
                    "cache_manifest",
                    "cache_row_index",
                    "proxy_crop_run",
                    "alias_manifest",
                ):
                    clips[-1].pop(legacy_cache_key)
        if scope == WORKFLOW_SCOPE_DOWNSTREAM:
            assert target.finalized_refined_detect_run is not None
            crop_row_intervals = _refined_clip_crop_row_intervals(
                target=target,
                refined_run_name=target.finalized_refined_detect_run,
                frame_intervals=frame_intervals,
            )
            for clip in clips:
                key = (int(clip["clip_index"]), str(clip["clip_id"]))
                crop_row_start, crop_row_stop = crop_row_intervals[key]
                clip["crop_row_start"] = crop_row_start
                clip["crop_row_stop"] = crop_row_stop
        recording_target = clipped_recording_target(
            target_id=target.target_id,
            recording_id=target.recording_id,
            recording_dir=target.recording_dir,
            analysis_zarr=target.analysis_zarr,
            work_units=work_units,
            expected_subject_count=target.expected_subject_count,
        )
        layout_work_units = {
            unit.work_unit_id: unit for unit in recording_target.work_units
        }
        target_payload: dict[str, Any] = {
            **target.to_json(),
            "target_label": target_label,
            "detection_plan_path": str(detection_plan_path),
            "native_detection_authority": authority.to_json(),
            "native_canonical_run_id": native_canonical_run,
            "canonical_refined_run_id": canonical_refined_run,
            "planned_registered_gate_group_path": planned_gate_group_path,
            "strict_storage_bundle_root": str(strict_storage_bundle),
            "collection_id": collection_id,
            "detect_quality_source_run": detect_quality_source_run,
            "detect_quality_source_group_path": (
                f"detect_collection_sources/{detect_quality_source_run}"
            ),
            "detect_quality_run": detect_quality_run,
            "detect_quality_group_path": f"detect_quality_runs/{detect_quality_run}",
            "cache_dir": str(cache_dir),
            "package_dir": str(package_dir),
            "global_mask_grid_manifest": str(global_mask_grid_manifest),
            "merged_proxy_crop_run": merged_proxy,
            "hybrid_crop_run": (
                hybrid_crop_run if scope == WORKFLOW_SCOPE_DOWNSTREAM else None
            ),
            "hybrid_supplemental_manifest": (
                str(
                    target.recording_dir
                    / "derived"
                    / "roi_cache"
                    / hybrid_crop_run
                    / (
                        f"{target.analysis_zarr.name.removesuffix('.zarr')}__"
                        f"{hybrid_crop_run}.supplemental.flat_roi_cache.json"
                    )
                )
                if scope == WORKFLOW_SCOPE_DOWNSTREAM
                else None
            ),
            "keypoint_run": keypoint_run,
            "keypoint_finalization_mapping_mode": (
                "direct_same_crop_row_ids"
                if scope == WORKFLOW_SCOPE_DOWNSTREAM
                else "identity_rebase"
            ),
            "refined_keypoint_run": refined_keypoint_run,
            "subject_mask_run": subject_mask_run,
            "refined_subject_mask_run": refined_subject_mask_run,
            "refined_subject_mask_draft_run": refined_subject_mask_draft_run,
            "subject_mask_quality_run": subject_mask_quality_run,
            "subject_mask_cache_run": subject_mask_cache_run,
            "subject_mask_bundle_id": subject_mask_bundle_id,
            "clips": clips,
            "registered_dish_geometry": {
                "gate_requirement": gate_requirement,
                "selection_run": geometry_selection_run,
                "gate_run": target_gate_run,
                "gate_materialization_planned": materialize_registered_gate,
                "selection_policy_id": policy_id,
            },
        }
        _refuse_output_collisions(
            target_payload,
            workflow_scope=scope,
            allow_existing_detections=resume_existing_detections,
        )
        if resume_existing_detections and scope != WORKFLOW_SCOPE_DOWNSTREAM:
            target_payload["detection_resume_preflight"] = [
                _validate_existing_detection_for_resume(
                    target=target,
                    workflow_id=workflow_id,
                    clip=clip,
                    binding=detection_binding,
                )
                for clip in clips
            ]
        target_payloads.append(target_payload)

        gate: tuple[str, ...] = ()
        if target_index >= max_active_targets:
            gate = (target_terminal_keys[target_index - max_active_targets],)

        gate_artifacts: tuple[str, ...] = ()
        if target_index >= max_active_targets:
            gate_artifacts = (
                target_terminal_artifacts[target_index - max_active_targets],
            )
        if scope == WORKFLOW_SCOPE_DOWNSTREAM:
            assert pose_binding is not None
            assert subject_binding is not None
            target_payload["detection_module"] = {
                "target_id": target.target_id,
                "source": "external_finalized_recording_refined_detection",
                "refined_group_path": (
                    f"refined_detect_runs/{target.finalized_refined_detect_run}"
                ),
                "authority_required": (
                    "complete_gated_immutable_recording_authority_v1"
                ),
                "clip_slice_index_published": False,
                "publication_partition": "complete_recording_snapshot",
            }
            downstream = _build_downstream_target_pipeline(
                workflow_id=workflow_id,
                repo=repo,
                repo_commit=repo_commit,
                registry_path=registry_path,
                run_root=run_root,
                target=target,
                target_safe=target_safe,
                target_payload=target_payload,
                pose_binding=pose_binding,
                subject_binding=subject_binding,
                subject_mask_coverage_class=subject_mask_coverage_class,
                subject_mask_component_coverage_key=(
                    subject_mask_component_coverage_key
                ),
                subject_mask_label_schema_id=subject_mask_label_schema_id,
                subject_mask_publication_profile=(subject_mask_publication_profile),
                encoded_mask_packages=encoded_mask_packages,
                gpu_array_concurrency=gpu_array_concurrency,
                mask_package_array_concurrency=mask_package_array_concurrency,
                keypoint_gpu_queue=keypoint_gpu_queue,
                subject_mask_gpu_queue=subject_mask_gpu_queue,
                cpu=cpu,
                final_cpu=final_cpu,
                import_cpu=import_cpu,
                cache_gpu=cache_gpu,
                upstream_job_keys=gate,
                required_artifacts=gate_artifacts,
            )
            jobs.extend(downstream.jobs)
            fragments.extend(downstream.fragments)
            target_terminal_keys.append(downstream.terminal_job_key)
            target_terminal_artifacts.append(downstream.terminal_artifact_key)
            continue
        native_module = build_native_detection_fragment(
            NativeDetectionFragmentInputs(
                workflow_id=workflow_id,
                family=FAMILY,
                target_id=target.target_id,
                recording_identity=authority.recording_identity,
                recording_dir=target.recording_dir,
                analysis_zarr=target.analysis_zarr,
                repo=repo,
                run_root=run_root,
                canonical_run_id=native_canonical_run,
                n_frames=authority.n_frames,
                source_width=authority.source_width,
                source_height=authority.source_height,
                source_frame_authority=authority.frame,
                source_pixel_authority=authority.pixel,
                producer_version=repo_commit,
                clips=tuple(
                    NativeDetectionClipSpec.from_plan_work_unit(
                        unit,
                        report_path=(
                            run_root
                            / "targets"
                            / target_safe
                            / "detection_reports"
                            / f"{unit['clip_id']}.json"
                        ),
                    )
                    for unit in work_units
                ),
                model=NativeDetectionModelSpec(
                    set_id=detection_binding.set_id,
                    run_id=detection_binding.run_id,
                    path=detection_binding.path,
                    sha256=detection_binding.sha256,
                ),
                detect_array_concurrency=detect_array_concurrency,
                resume_existing_artifacts=resume_existing_detections,
                upstream_job_keys=gate,
                required_artifacts=gate_artifacts,
            )
        )
        raw_module = RawDetectionWorkflowModule(
            fragment=native_module.fragment,
            outputs=RawDetectionFragmentOutputs(
                target_id=target.target_id,
                raw_detection_group_paths=native_module.outputs.artifact_group_paths,
                terminal_job_key=native_module.outputs.terminal_job_key,
                artifact_key=native_module.outputs.artifact_key,
            ),
        )
        detection_inputs = DetectionFragmentInputs(
            workflow_id=workflow_id,
            family=FAMILY,
            target_label=target_label,
            target=recording_target,
            repo=repo,
            run_root=run_root,
            detection_plan_path=detection_plan_path,
            collection_id=collection_id,
            quality_source_run=detect_quality_source_run,
            quality_run=detect_quality_run,
            work_units=tuple(
                DetectionWorkUnitSpec.from_mapping(
                    clip,
                    work_unit=layout_work_units[str(clip["work_unit_id"])],
                )
                for clip in clips
            ),
            model=DetectionModelSpec(
                set_id=detection_binding.set_id,
                run_id=detection_binding.run_id,
                path=detection_binding.path,
                sha256=detection_binding.sha256,
            ),
            resume_existing_detections=resume_existing_detections,
            detect_array_concurrency=detect_array_concurrency,
            refine_bundle_concurrency=detect_refine_bundle_concurrency,
        )
        target_payload["native_detection_module"] = native_module.outputs.to_json()
        if canonical_recording_pipeline:
            jobs.extend(native_module.fragment.jobs)
            fragments.append(native_module.fragment)
            assert canonical_refined_run is not None
            postprocess_upstream = (native_module.outputs.terminal_job_key,)
            postprocess_artifacts = (native_module.outputs.artifact_key,)
            detection_module_fragments = [native_module.fragment]
            if materialize_registered_gate:
                assert geometry_selection_run is not None
                assert target_gate_run is not None
                gate_module = build_registered_detection_gate_fragment(
                    RegisteredDetectionGateFragmentInputs(
                        workflow_id=workflow_id,
                        family=FAMILY,
                        target=recording_target,
                        repo=repo,
                        run_root=run_root,
                        source_detection_group_path=(
                            native_module.outputs.canonical_group_path
                        ),
                        selection_run=geometry_selection_run,
                        output_run=target_gate_run,
                        upstream_job_keys=(native_module.outputs.terminal_job_key,),
                        required_artifacts=(native_module.outputs.artifact_key,),
                    )
                )
                jobs.extend(gate_module.fragment.jobs)
                fragments.append(gate_module.fragment)
                detection_module_fragments.append(gate_module.fragment)
                postprocess_upstream = (gate_module.outputs.terminal_job_key,)
                postprocess_artifacts = (gate_module.outputs.artifact_key,)
                target_payload["registered_detection_gate_module"] = (
                    gate_module.outputs.to_json()
                )
            postprocess = build_recording_detection_postprocess_fragment(
                RecordingDetectionPostprocessInputs(
                    workflow_id=workflow_id,
                    family=FAMILY,
                    target=recording_target,
                    repo=repo,
                    run_root=run_root,
                    source_detect_run=native_canonical_run,
                    quality_run=detect_quality_run,
                    refined_run=canonical_refined_run,
                    registered_gate_requirement=gate_requirement,
                    registered_gate_run=target_gate_run,
                    selection_policy_id=policy_id,
                    require_active_canonical_source=True,
                    source_publication_receipt=(
                        native_module.outputs.publication_receipt_path
                    ),
                    upstream_job_keys=postprocess_upstream,
                    required_artifacts=postprocess_artifacts,
                )
            )
            jobs.extend(postprocess.fragment.jobs)
            fragments.append(postprocess.fragment)
            target_payload["canonical_refined_run_id"] = canonical_refined_run
            target_payload["detect_quality_source_group_path"] = (
                f"detect_runs/{native_canonical_run}"
            )
            target_payload["detect_quality_group_path"] = (
                f"detect_quality_runs/{detect_quality_run}"
            )
            if scope == WORKFLOW_SCOPE_DETECTION:
                canonical_outputs = {
                    **postprocess.outputs.to_json(),
                    "publication_authority": "canonical_recording_refined_snapshot",
                    "clip_slice_index_published": False,
                }
                target_payload["detection_module"] = canonical_outputs
                detection_scope_outputs.append(canonical_outputs)
                target_terminal_keys.append(postprocess.outputs.terminal_job_key)
                target_terminal_artifacts.append(postprocess.outputs.artifact_key)
                continue

            collection_key = f"registered_refined_collection:{target_safe}"
            collection_receipt = (
                run_root / "detection_collections" / f"{target_safe}.json"
            )
            collection_command = [
                "scripts/py",
                "-m",
                "fisheye.utils.finalize_registered_clipped_refined_collection",
                "--plan",
                str(run_root / "plan.json"),
                "--target-id",
                target.target_id,
                "--analysis-zarr",
                str(target.analysis_zarr),
                "--collection-id",
                collection_id,
                "--refined-run",
                canonical_refined_run,
                "--recording-frame-index",
                str(target.recording_dir / "recording_frame_index.parquet"),
                "--registered-gate-requirement",
                gate_requirement,
                "--result-json",
                str(collection_receipt),
            ]
            if target_gate_run is not None:
                collection_command.extend(("--registered-gate-run", target_gate_run))
            collection_job = _job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=collection_key,
                stage="refined_detect_collection",
                command=collection_command,
                resources=final_cpu,
                upstream=(postprocess.outputs.terminal_job_key,),
                expected_outputs=(
                    target.analysis_zarr
                    / "experiment_index"
                    / "finalized_runs"
                    / collection_id
                    / "zarr.json",
                    collection_receipt,
                ),
            )
            jobs.append(collection_job)
            collection_artifact = f"finalized_refined_detection:{target_safe}"
            collection_fragment = LsfWorkflowFragment(
                fragment_id=f"registered_refined_collection:{target_safe}",
                jobs=(collection_job,),
                requires=(postprocess.outputs.artifact_key,),
                provides=(collection_artifact,),
                metadata={
                    "module": "registered_refined_clipped_collection",
                    "source": postprocess.outputs.to_json(),
                    "collection_id": collection_id,
                    "slice_authority": "canonical_recording_refined_run",
                    "selector_activation": False,
                },
            )
            fragments.append(collection_fragment)
            detection_outputs = DetectionFragmentOutputs(
                target_id=target.target_id,
                collection_id=collection_id,
                raw_detection_group_paths=(f"detect_runs/{native_canonical_run}",),
                quality_source_group_path=f"detect_runs/{native_canonical_run}",
                quality_group_path=f"detect_quality_runs/{detect_quality_run}",
                refined_detection_group_paths=(
                    f"refined_detect_runs/{canonical_refined_run}",
                ),
                finalized_collection_group_path=(
                    f"experiment_index/finalized_runs/{collection_id}"
                ),
                terminal_job_key=collection_key,
                artifact_key=collection_artifact,
            )
            downstream_detection_terminal = collection_key
            downstream_detection_artifact = collection_artifact
            downstream_detection_authority = {
                "postprocess": postprocess.outputs.to_json(),
                "collection_id": collection_id,
                "collection_receipt": str(collection_receipt),
            }
            detection_module = DetectionWorkflowModule(
                fragments=tuple(
                    detection_module_fragments
                    + [postprocess.fragment, collection_fragment]
                ),
                raw_outputs=raw_module.outputs,
                outputs=detection_outputs,
            )
        else:
            detection_module = build_detection_fragment(
                detection_inputs,
                raw_module=raw_module,
            )
            jobs.extend(detection_module.jobs)
            fragments.extend(detection_module.fragments)
            detection_outputs = detection_module.outputs
        target_payload["detection_module"] = detection_outputs.to_json()
        if scope == WORKFLOW_SCOPE_DETECTION:
            target_terminal_keys.append(detection_outputs.terminal_job_key)
            target_terminal_artifacts.append(detection_outputs.artifact_key)
            continue

        assert pose_binding is not None
        assert subject_binding is not None
        storage_modules = None
        if not modern_registered_pipeline:
            strict_clip_inputs = tuple(
                StrictClipRefinedDetectionInput(
                    clip_index=int(clip["clip_index"]),
                    clip_id=str(clip["clip_id"]),
                    archive=(
                        strict_storage_bundle
                        / f"clip_{int(clip['clip_index']):06d}_{clip['clip_id']}"
                        / "refined.zarr"
                    ),
                    run_id=f"strict_{clip['refined_detect_run']}",
                )
                for clip in clips
            )
            storage_modules = build_clipped_detection_storage_fragments(
                ClippedDetectionEvidenceInputs(
                    workflow_id=workflow_id,
                    family=FAMILY,
                    target_id=target.target_id,
                    analysis_zarr=target.analysis_zarr,
                    recording_canonical_archive=target.analysis_zarr,
                    recording_canonical_run_id=native_canonical_run,
                    recording_identity=authority.recording_identity,
                    detection_plan_path=detection_plan_path,
                    collection_id=collection_id,
                    recording_dir=target.recording_dir,
                    bundle_root=strict_storage_bundle,
                    clips=tuple(
                        ClipDetectionEvidenceInput(
                            clip_index=int(clip["clip_index"]),
                            clip_id=str(clip["clip_id"]),
                            source_detect_group_path=str(clip["detect_group_path"]),
                            source_refined_group_path=str(
                                clip["refined_detect_group_path"]
                            ),
                            canonical_run_id=f"strict_{clip['detect_run']}",
                            refined_run_id=f"strict_{clip['refined_detect_run']}",
                        )
                        for clip in clips
                    ),
                    repo=repo,
                    run_root=run_root,
                    max_concurrent=detect_refine_bundle_concurrency,
                    upstream_job_keys=(detection_outputs.terminal_job_key,),
                    required_artifacts=(detection_outputs.artifact_key,),
                ),
                ClippedStorageFinalizationInputs(
                    workflow_id=workflow_id,
                    family=FAMILY,
                    target_id=target.target_id,
                    analysis_zarr=target.analysis_zarr,
                    canonical_archive=target.analysis_zarr,
                    canonical_run_id=native_canonical_run,
                    clips=strict_clip_inputs,
                    clipped_binding_path=(
                        strict_storage_bundle / "clipped_refined_detection_binding.json"
                    ),
                    bundle_root=strict_storage_bundle,
                    refined_run_id=f"refined_detection_snapshot_{target_label}",
                    refined_lineage_id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"palette:refined-lineage:{authority.recording_identity}",
                        )
                    ),
                    refined_snapshot_id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            "palette:refined-snapshot:"
                            f"{authority.recording_identity}:{target_label}",
                        )
                    ),
                    crop_run_id=f"crop_v2_{target_label}",
                    recording_identity=authority.recording_identity,
                    crop_purpose="keypoints_subject_masks",
                    roi_width=512,
                    roi_height=512,
                    camera_id=authority.camera_serial,
                    repo=repo,
                    run_root=run_root,
                ),
            )
            jobs.extend(storage_modules.evidence.fragment.jobs)
            jobs.extend(storage_modules.storage.fragment.jobs)
            fragments.extend(
                (storage_modules.evidence.fragment, storage_modules.storage.fragment)
            )
            target_payload["strict_detection_storage"] = {
                "evidence": storage_modules.evidence.outputs.to_json(),
                "storage": storage_modules.storage.outputs.to_json(),
            }
            downstream_detection_terminal = (
                storage_modules.storage.outputs.terminal_job_key
            )
            downstream_detection_artifact = (
                storage_modules.storage.outputs.crop_artifact_key
            )
            downstream_detection_authority = storage_modules.storage.outputs.to_json()
        else:
            target_payload["strict_detection_storage"] = None
        cache_array_key = f"cache_array:{target_safe}"
        cache_tasks: list[LsfExecutionTask] = []
        for bundle_index, start in enumerate(range(0, len(clips), cache_bundle_size)):
            bundle_clips = clips[start : start + cache_bundle_size]
            cache_key = f"cache:{target_safe}:{bundle_index:02d}"
            cache_command = [
                "bash",
                "scripts/submit_clipped_collection_flat_roi_cache_bundle_bsub.sh",
                "--zarr",
                str(target.analysis_zarr),
                "--collection-id",
                detection_outputs.collection_id,
                "--recording-frame-index",
                str(target.recording_dir / "recording_frame_index.parquet"),
                "--public-cache-dir",
                str(cache_dir),
                "--run-id",
                f"{target_label}_{bundle_index:02d}",
                "--run-label",
                f"roi_cache_{target_label}",
                "--log-dir",
                str(run_root / "cache_jobs" / target_safe),
                "--max-workers",
                str(len(bundle_clips)),
                "--gpus",
                "0",
                "--run-direct",
                "--sha256",
            ]
            for clip in bundle_clips:
                cache_command.extend(["--clip-id", str(clip["clip_id"])])
            cache_tasks.append(
                _execution_task(
                    run_root=run_root,
                    task_key=cache_key,
                    stage="roi_cache",
                    command=cache_command,
                    expected_outputs=tuple(
                        Path(str(clip["cache_manifest"])) for clip in bundle_clips
                    ),
                    array_indexed=True,
                )
            )
        jobs.append(
            _task_group_job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=cache_array_key,
                stage="roi_cache",
                tasks=cache_tasks,
                mode=LsfExecutionMode.ARRAY,
                max_concurrent=cache_array_concurrency,
                resources=cache_gpu,
                upstream=(downstream_detection_terminal,),
            )
        )

        proxy_key = f"proxy:{target_safe}"
        proxy_commands: list[list[str]] = []
        for clip in clips:
            proxy_commands.append(
                [
                    "scripts/py",
                    "-m",
                    "fisheye.utils.create_clipped_collection_proxy_crop_run",
                    str(target.analysis_zarr),
                    str(clip["cache_manifest"]),
                    "--proxy-run",
                    str(clip["proxy_crop_run"]),
                    "--alias-manifest",
                    str(clip["alias_manifest"]),
                    "--json",
                ]
            )
        jobs.append(
            _job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=proxy_key,
                stage="proxy_crop",
                command=_chain(proxy_commands),
                resources=cpu,
                upstream=(cache_array_key,),
                expected_outputs=tuple(
                    Path(str(clip["alias_manifest"])) for clip in clips
                ),
            )
        )

        keypoint_array_key = f"keypoints_array:{target_safe}"
        subject_mask_array_key = f"subject_masks_array:{target_safe}"
        keypoint_tasks: list[LsfExecutionTask] = []
        mask_tasks: list[LsfExecutionTask] = []
        for clip in clips:
            clip_id = str(clip["clip_id"])
            keypoint_key = f"keypoints:{target_safe}:{clip_id}"
            keypoint_command = [
                "scripts/py",
                "-m",
                "fisheye.utils.run_keypoints_with_registry_model",
                "--recording-dir",
                str(target.recording_dir),
                "--output",
                str(target.analysis_zarr),
                "--registry",
                str(registry_path),
                "--set-id",
                pose_binding.set_id,
                "--model-run-id",
                pose_binding.run_id,
                "--require-unique",
                "--run-name",
                str(clip["keypoint_shard_run"]),
                "--output-parent",
                "keypoint_shard_runs",
                "--coordinate-contract-mode",
                "legacy_noncanonical",
                "--crop-run",
                str(clip["proxy_crop_run"]),
                "--pose-schema",
                "traditional_v2",
                "--batch-size",
                "256",
                "--device",
                "0",
                "--roi-cache-manifest",
                str(clip["alias_manifest"]),
                "--stage-roi-cache-to-scratch",
                "--keypoint-roi-shard-rows",
                "131072",
                "--keypoint-frame-shard-rows",
                "131072",
                "--progress-jsonl",
                str(run_root / "progress" / f"keypoints_{target_safe}_{clip_id}.jsonl"),
            ]
            keypoint_tasks.append(
                _execution_task(
                    run_root=run_root,
                    task_key=keypoint_key,
                    stage="keypoints",
                    command=keypoint_command,
                    expected_outputs=(
                        target.analysis_zarr
                        / "keypoint_shard_runs"
                        / str(clip["keypoint_shard_run"])
                        / "zarr.json",
                    ),
                    array_indexed=True,
                )
            )

            mask_key = f"subject_masks:{target_safe}:{clip_id}"
            mask_staging_dir = (
                f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}_"
                f"{RUNTIME_JOB_INDEX_TOKEN}/palette_subject_mask_roi_cache"
            )
            mask_worker_receipt = (
                run_root / "receipts" / f"subject_masks_{target_safe}_{clip_id}.json"
            )
            mask_command = [
                "scripts/py",
                "-m",
                "fisheye.cluster.subject_masks.staged_inference",
                "--roi-cache-staging-dir",
                mask_staging_dir,
                "--worker-receipt-json",
                str(mask_worker_receipt),
                str(target.analysis_zarr),
                "--resolve-model-from-registry",
                "--registry",
                str(registry_path),
                "--model-set-id",
                subject_binding.set_id,
                "--model-run-id",
                subject_binding.run_id,
                "--model-coverage-class",
                subject_mask_coverage_class,
                "--model-component-coverage-key",
                subject_mask_component_coverage_key,
                "--model-label-schema-id",
                subject_mask_label_schema_id,
                "--model-require-unique",
                "--run-name",
                str(clip["subject_mask_shard_run"]),
                "--output-parent",
                "subject_mask_shard_runs",
                "--crop-run",
                str(clip["proxy_crop_run"]),
                "--source-collection-id",
                detection_outputs.collection_id,
                "--source-collection-path",
                detection_outputs.finalized_collection_group_path,
                "--source-clip-id",
                clip_id,
                "--source-clip-index",
                str(clip["clip_index"]),
                "--source-work-unit-id",
                str(clip["work_unit_id"]),
                "--source-roi-cache-alias-manifest",
                str(clip["alias_manifest"]),
                "--source-roi-cache-row-index-path",
                str(clip["cache_row_index"]),
                "--roi-cache-manifest",
                str(clip["alias_manifest"]),
                "--roi-cache-policy",
                "never",
                "--batch-size",
                "128",
                "--device",
                "0",
                "--mask-probs-dtype",
                "uint8",
                "--mask-probs-chunk-rois",
                "32",
                "--mask-probs-shard-rois",
                "2048",
                "--no-write-masks-roi",
                "--async-output",
                "--output-queue-size",
                "2",
                "--no-progress",
                "--defer-registry-status",
            ]
            mask_tasks.append(
                _execution_task(
                    run_root=run_root,
                    task_key=mask_key,
                    stage="subject_mask_inference",
                    command=mask_command,
                    expected_outputs=(
                        target.analysis_zarr
                        / "subject_mask_shard_runs"
                        / str(clip["subject_mask_shard_run"])
                        / "zarr.json",
                        mask_worker_receipt,
                    ),
                    array_indexed=True,
                )
            )
        jobs.append(
            _task_group_job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=keypoint_array_key,
                stage="keypoints",
                tasks=keypoint_tasks,
                mode=LsfExecutionMode.ARRAY,
                max_concurrent=gpu_array_concurrency,
                resources=gpu,
                upstream=(proxy_key,),
            )
        )
        jobs.append(
            _task_group_job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=subject_mask_array_key,
                stage="subject_mask_inference",
                tasks=mask_tasks,
                mode=LsfExecutionMode.ARRAY,
                max_concurrent=gpu_array_concurrency,
                resources=gpu,
                upstream=(proxy_key,),
            )
        )

        keypoint_finalize_key = f"keypoint_finalize:{target_safe}"
        merge_proxy = [
            "scripts/py",
            "-m",
            "fisheye.utils.merge_clipped_proxy_crop_runs",
            str(target.analysis_zarr),
            "--output-run",
            merged_proxy,
            "--json",
        ]
        for clip in clips:
            merge_proxy.extend(["--source-crop-run", str(clip["proxy_crop_run"])])
        finalize_keypoints = [
            "scripts/py",
            "-m",
            "fisheye.utils.finalize_keypoint_shards",
            str(target.analysis_zarr),
            "--target-crop-run",
            merged_proxy,
            "--output-run",
            keypoint_run,
            "--json",
        ]
        for clip in clips:
            finalize_keypoints.extend(["--shard-run", str(clip["keypoint_shard_run"])])
        jobs.append(
            _job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=keypoint_finalize_key,
                stage="keypoint_finalize",
                command=_chain((merge_proxy, finalize_keypoints)),
                resources=final_cpu,
                upstream=(keypoint_array_key,),
                expected_outputs=(
                    target.analysis_zarr / "crop_runs" / merged_proxy / "zarr.json",
                    target.analysis_zarr
                    / "keypoints_runs"
                    / keypoint_run
                    / "zarr.json",
                ),
            )
        )
        keypoint_refine_key = f"keypoint_refine:{target_safe}"
        keypoint_refine = [
            "scripts/py",
            "-m",
            "fisheye.refinement.refine_keypoints",
            str(target.analysis_zarr),
            "--keypoint-run",
            keypoint_run,
            "--run-name",
            refined_keypoint_run,
            "--chunk-size",
            "2048",
            "--scheduler",
            "threads",
            "--num-workers",
            "4",
            "--no-post-audit",
        ]
        jobs.append(
            _job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=keypoint_refine_key,
                stage="keypoint_refine",
                command=keypoint_refine,
                resources=cpu,
                upstream=(keypoint_finalize_key,),
                expected_outputs=(
                    target.analysis_zarr
                    / "refined_keypoints_runs"
                    / refined_keypoint_run
                    / "zarr.json",
                ),
            )
        )

        mask_grid_key: str | None = None
        if encoded_mask_packages:
            mask_grid_key = f"mask_grid:{target_safe}"
            mask_grid_command = [
                "scripts/py",
                "-m",
                "fisheye.utils.prepare_refined_subject_mask_chunk_grid",
                "--zarr",
                str(target.analysis_zarr),
                "--crop-run",
                merged_proxy,
                "--output-manifest",
                str(global_mask_grid_manifest),
                "--mask-label",
                "subject_body",
                "--mask-label",
                "eye_left",
                "--mask-label",
                "eye_right",
                "--mask-label",
                "swim_bladder",
                "--mask-height",
                "512",
                "--mask-width",
                "512",
                "--dense-mask-row-chunk",
                "128",
                "--json",
            ]
            jobs.append(
                _job(
                    workflow_id=workflow_id,
                    repo=repo,
                    run_root=run_root,
                    job_key=mask_grid_key,
                    stage="subject_mask_global_chunk_grid",
                    command=mask_grid_command,
                    resources=cpu,
                    upstream=(keypoint_finalize_key,),
                    expected_outputs=(global_mask_grid_manifest,),
                )
            )

        package_array_key = f"mask_package_array:{target_safe}"
        package_tasks: list[LsfExecutionTask] = []
        for clip in clips:
            clip_id = str(clip["clip_id"])
            package_key = f"mask_package:{target_safe}:{clip_id}"
            package_command = [
                "scripts/py",
                "-m",
                "fisheye.utils.finalize_subject_mask_clip_package",
                "--source-zarr",
                str(target.analysis_zarr),
                "--subject-shard-run",
                str(clip["subject_mask_shard_run"]),
                "--target-crop-run",
                merged_proxy,
                "--refined-run",
                str(clip["refined_mask_package_run"]),
                "--package-path",
                str(clip["package_path"]),
                "--component",
                "subject_body",
                "--component",
                "eyes_union",
                "--component",
                "swim_bladder",
                "--chunk-size",
                "256",
                "--metric-level",
                "cheap",
                "--mask-storage",
                "dense_and_bitpacked",
                "--dense-mask-row-chunk",
                "128",
                "--execution-backend",
                "process_shards",
                "--num-workers",
                "8",
                "--postcompute-backend",
                "process_shards",
                "--postcompute-num-workers",
                "8",
                "--postcompute-chunk-size",
                "256",
                "--assignment-keypoint-group",
                "refined_keypoints_runs",
                "--assignment-keypoints-run",
                refined_keypoint_run,
                "--no-write-component-contours",
                "--require-production-proof",
                "--json",
            ]
            if encoded_mask_packages:
                package_command.extend(
                    [
                        "--global-mask-grid-manifest",
                        str(global_mask_grid_manifest),
                        "--encoded-mask-copy-workers",
                        "8",
                    ]
                )
            if (
                subject_mask_publication_profile
                == SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED
            ):
                package_command.extend(
                    [
                        "--publication-evidence-producer-commit",
                        repo_commit,
                        "--work-unit-id",
                        str(clip["work_unit_id"]),
                        "--work-unit-index",
                        str(clip["work_unit_index"]),
                        "--source-clip-id",
                        clip_id,
                        "--source-clip-index",
                        str(clip["clip_index"]),
                        "--global-frame-start",
                        str(clip["frame_start"]),
                        "--global-frame-stop",
                        str(clip["frame_stop"]),
                        "--quality-compute-workers",
                        "4",
                    ]
                )
            package_tasks.append(
                _execution_task(
                    run_root=run_root,
                    task_key=package_key,
                    stage="subject_mask_refine_package",
                    command=package_command,
                    expected_outputs=(Path(str(clip["package_path"])),),
                    array_indexed=True,
                )
            )
        package_upstream = [subject_mask_array_key, keypoint_refine_key]
        if mask_grid_key is not None:
            package_upstream.append(mask_grid_key)
        jobs.append(
            _task_group_job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=package_array_key,
                stage="subject_mask_refine_package",
                tasks=package_tasks,
                mode=LsfExecutionMode.ARRAY,
                max_concurrent=mask_package_array_concurrency,
                resources=final_cpu,
                upstream=tuple(package_upstream),
            )
        )

        mask_import_key = f"mask_import:{target_safe}"
        mask_import = [
            "scripts/py",
            "-m",
            "fisheye.utils.import_refined_subject_mask_clip_packages",
            "--zarr",
            str(target.analysis_zarr),
            "--output-run",
            refined_subject_mask_draft_run,
            "--expected-target-crop-run",
            merged_proxy,
            "--array-copy-workers",
            "8",
            "--encoded-copy-workers",
            "32",
            "--require-production-proof",
            "--json",
        ]
        for clip in clips:
            mask_import.extend(["--package", str(clip["package_path"])])
        if (
            subject_mask_publication_profile
            == SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK
        ):
            jobs.append(
                _job(
                    workflow_id=workflow_id,
                    repo=repo,
                    run_root=run_root,
                    job_key=mask_import_key,
                    stage="subject_mask_collection_import",
                    command=mask_import,
                    resources=import_cpu,
                    upstream=(package_array_key,),
                    expected_outputs=(
                        target.analysis_zarr
                        / "refined_subject_masks_runs"
                        / refined_subject_mask_draft_run
                        / "zarr.json",
                    ),
                )
            )
        else:
            mask_import_key = package_array_key

        mask_publish_key = f"mask_publish:{target_safe}"
        mask_publish_output = (
            f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
            "palette_subject_mask_bundle_outputs"
        )
        mask_quality_scratch = (
            f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
            "palette_subject_mask_quality"
        )
        receipt_composed = (
            subject_mask_publication_profile
            == SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED
        )
        mask_publish = [
            "scripts/py",
            "-m",
            (
                "fisheye.cluster.subject_masks.publish_receipt_composed_bundle"
                if receipt_composed
                else "fisheye.cluster.subject_masks.publish_recording_bundle"
            ),
            "--analysis-zarr",
            str(target.analysis_zarr),
            "--crop-run",
            merged_proxy,
            "--raw-run",
            subject_mask_run,
            "--refined-run",
            refined_subject_mask_run,
            "--quality-run",
            subject_mask_quality_run,
            "--cache-run",
            subject_mask_cache_run,
            "--bundle-id",
            subject_mask_bundle_id,
            "--local-output-root",
            mask_publish_output,
            "--quality-scratch-root",
            mask_quality_scratch,
            "--json",
        ]
        if receipt_composed:
            mask_publish.extend(["--producer-commit", repo_commit])
            for clip in clips:
                mask_publish.extend(["--refined-package", str(clip["package_path"])])
        else:
            mask_publish.extend(
                [
                    "--draft-zarr",
                    str(target.analysis_zarr),
                    "--raw-draft-parent",
                    "subject_mask_shard_runs",
                    "--refined-draft-run",
                    refined_subject_mask_draft_run,
                ]
            )
        for clip in clips:
            mask_publish.extend(
                ["--raw-draft-run", str(clip["subject_mask_shard_run"])]
            )
        jobs.append(
            _job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=mask_publish_key,
                stage="subject_mask_collection_publication",
                command=mask_publish,
                resources=import_cpu,
                upstream=(mask_import_key,),
                cleanup_paths=(mask_publish_output, mask_quality_scratch),
                expected_outputs=(
                    target.analysis_zarr
                    / "subject_mask_bundle_runs"
                    / subject_mask_bundle_id
                    / "zarr.json",
                ),
            )
        )

        validation_key = f"validate:{target_safe}"
        validation_report = run_root / "validation" / f"{target_safe}.json"
        validation_command = [
            "scripts/py",
            "-m",
            "fisheye.cluster.clipped_inference_validate",
            "--plan",
            str(run_root / "plan.json"),
            "--target-id",
            target.target_id,
            "--output-json",
            str(validation_report),
        ]
        jobs.append(
            _job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key=validation_key,
                stage="validation",
                command=validation_command,
                resources=final_cpu,
                upstream=(mask_publish_key,),
                expected_outputs=(validation_report,),
            )
        )
        crop_cache_artifact = f"crop_roi_cache:{target_safe}"
        raw_keypoints_artifact = f"raw_keypoints:{target_safe}"
        refined_keypoints_artifact = f"refined_keypoints:{target_safe}"
        raw_masks_artifact = f"raw_subject_masks:{target_safe}"
        refined_masks_artifact = f"refined_subject_masks:{target_safe}"
        validated_artifact = f"validated_analysis:{target_safe}"
        target_terminal_keys.append(validation_key)
        target_terminal_artifacts.append(validated_artifact)
        job_by_key = {job.job_key: job for job in jobs}
        mask_finalize_keys = list(
            dict.fromkeys((package_array_key, mask_import_key, mask_publish_key))
        )
        if mask_grid_key is not None:
            mask_finalize_keys.insert(0, mask_grid_key)
        fragments.extend(
            (
                LsfWorkflowFragment(
                    fragment_id=f"crop_roi_cache:{target_safe}",
                    jobs=(job_by_key[cache_array_key], job_by_key[proxy_key]),
                    requires=(downstream_detection_artifact,),
                    provides=(crop_cache_artifact,),
                    metadata={
                        "module": "crop_roi_cache",
                        "target_id": target.target_id,
                        "recording_layout": "clipped_collection",
                        "crop_authority": "proxy_crop_runs_bound_to_flat_roi_cache",
                        "finalized_detection_authority": (
                            downstream_detection_authority
                        ),
                        "terminal_job_key": proxy_key,
                    },
                ),
                LsfWorkflowFragment(
                    fragment_id=f"keypoints:{target_safe}",
                    jobs=(
                        job_by_key[keypoint_array_key],
                        job_by_key[keypoint_finalize_key],
                        job_by_key[keypoint_refine_key],
                    ),
                    requires=(crop_cache_artifact,),
                    provides=(raw_keypoints_artifact, refined_keypoints_artifact),
                    metadata={
                        "module": "keypoints",
                        "target_id": target.target_id,
                        "recording_layout": "clipped_collection",
                        "terminal_job_key": keypoint_refine_key,
                    },
                ),
                LsfWorkflowFragment(
                    fragment_id=f"subject_mask_inference:{target_safe}",
                    jobs=(job_by_key[subject_mask_array_key],),
                    requires=(crop_cache_artifact,),
                    provides=(raw_masks_artifact,),
                    metadata={
                        "module": "subject_mask_inference",
                        "target_id": target.target_id,
                        "recording_layout": "clipped_collection",
                        "assignment_keypoints_required": False,
                        "terminal_job_key": subject_mask_array_key,
                    },
                ),
                LsfWorkflowFragment(
                    fragment_id=f"subject_mask_refinement:{target_safe}",
                    jobs=tuple(job_by_key[key] for key in mask_finalize_keys),
                    requires=(raw_masks_artifact, refined_keypoints_artifact),
                    provides=(refined_masks_artifact,),
                    metadata={
                        "module": "subject_mask_refinement",
                        "target_id": target.target_id,
                        "recording_layout": "clipped_collection",
                        "terminal_job_key": mask_publish_key,
                    },
                ),
                LsfWorkflowFragment(
                    fragment_id=f"analysis_validation:{target_safe}",
                    jobs=(job_by_key[validation_key],),
                    requires=(refined_keypoints_artifact, refined_masks_artifact),
                    provides=(validated_artifact,),
                    metadata={
                        "module": "analysis_validation",
                        "target_id": target.target_id,
                        "detection_inputs": detection_outputs.to_json(),
                        "terminal_job_key": validation_key,
                    },
                ),
            )
        )

    if scope == WORKFLOW_SCOPE_DOWNSTREAM:
        scope_jobs = tuple(job for fragment in fragments for job in fragment.jobs)
        workflow = compose_lsf_workflow(
            workflow_id=workflow_id,
            family=FAMILY,
            fragments=tuple(fragments),
            metadata={
                "workflow_scope": "downstream_from_finalized_detection",
                "target_count": len(targets),
                "clip_count": sum(len(target["clips"]) for target in target_payloads),
                "publication_authority": ("recording_level_hybrid_crop_provider"),
                "clip_partitions_are_scheduler_work_only": True,
                "selector_activation": False,
                "model_bindings_are_exact": True,
                "gpu_queues": {
                    "keypoints": keypoint_gpu_queue,
                    "subject_masks": subject_mask_gpu_queue,
                },
                "scheduler_submission_count": len(scope_jobs),
                "execution_task_count": sum(
                    len(job.execution_group.tasks) if job.execution_group else 1
                    for job in scope_jobs
                ),
                "array_submission_count": sum(
                    1
                    for job in scope_jobs
                    if job.execution_group is not None
                    and job.execution_group.mode is LsfExecutionMode.ARRAY
                ),
            },
        )
        assert pose_binding is not None
        assert subject_binding is not None
        return ClippedInferencePlan(
            run_label=label,
            workflow_id=workflow_id,
            workflow_scope=scope,
            repo=repo,
            palette_commit=repo_commit,
            registry=registry_path,
            run_root=run_root,
            targets=tuple(targets),
            target_plans=tuple(target_payloads),
            model_bindings={
                "pose": pose_binding,
                "subject_masks": subject_binding,
            },
            max_active_targets=max_active_targets,
            cleanup_nrs_after_success=False,
            resume_existing_detections=False,
            encoded_mask_packages=encoded_mask_packages,
            subject_mask_publication_profile=subject_mask_publication_profile,
            detect_array_concurrency=int(detect_array_concurrency),
            gpu_array_concurrency=int(gpu_array_concurrency),
            cache_array_concurrency=int(cache_array_concurrency),
            mask_package_array_concurrency=int(mask_package_array_concurrency),
            detect_refine_bundle_concurrency=int(detect_refine_bundle_concurrency),
            registered_gate_requirement=gate_requirement,
            registered_gate_run=gate_run,
            selection_policy_id=policy_id,
            lsf_workflow=workflow,
        )

    if scope == WORKFLOW_SCOPE_DETECTION:
        scope_jobs = tuple(job for fragment in fragments for job in fragment.jobs)
        workflow = compose_lsf_workflow(
            workflow_id=workflow_id,
            family=FAMILY,
            fragments=tuple(fragments),
            metadata={
                "workflow_scope": "detection_only",
                "target_count": len(targets),
                "publication_authority": "canonical_recording_refined_snapshot",
                "clip_slice_indexes_published": False,
                "outputs": detection_scope_outputs,
                "scheduler_submission_count": len(scope_jobs),
                "execution_task_count": sum(
                    len(job.execution_group.tasks) if job.execution_group else 1
                    for job in scope_jobs
                ),
                "array_submission_count": sum(
                    1 for job in scope_jobs if job.execution_group is not None
                ),
            },
        )
        return ClippedInferencePlan(
            run_label=label,
            workflow_id=workflow_id,
            workflow_scope=scope,
            repo=repo,
            palette_commit=repo_commit,
            registry=registry_path,
            run_root=run_root,
            targets=tuple(targets),
            target_plans=tuple(target_payloads),
            model_bindings={"detection": detection_binding},
            max_active_targets=max_active_targets,
            cleanup_nrs_after_success=False,
            resume_existing_detections=resume_existing_detections,
            encoded_mask_packages=False,
            subject_mask_publication_profile=subject_mask_publication_profile,
            detect_array_concurrency=int(detect_array_concurrency),
            gpu_array_concurrency=int(gpu_array_concurrency),
            cache_array_concurrency=int(cache_array_concurrency),
            mask_package_array_concurrency=int(mask_package_array_concurrency),
            detect_refine_bundle_concurrency=int(detect_refine_bundle_concurrency),
            registered_gate_requirement=gate_requirement,
            registered_gate_run=gate_run,
            selection_policy_id=policy_id,
            lsf_workflow=workflow,
        )

    campaign_job_start = len(jobs)
    registry_key = "registry_finalize"
    registry_report = run_root / "registry" / "reconcile.json"
    registry_command = [
        "scripts/py",
        "-m",
        "fisheye.cluster.clipped_inference_registry_finalize",
        "--plan",
        str(run_root / "plan.json"),
        "--output-json",
        str(registry_report),
    ]
    jobs.append(
        _job(
            workflow_id=workflow_id,
            repo=repo,
            run_root=run_root,
            job_key=registry_key,
            stage="registry_finalize",
            command=registry_command,
            resources=cpu,
            upstream=tuple(target_terminal_keys),
            expected_outputs=(registry_report,),
        )
    )
    if cleanup_nrs_after_success:
        cleanup_report = run_root / "cleanup" / "nrs_cleanup.json"
        jobs.append(
            _job(
                workflow_id=workflow_id,
                repo=repo,
                run_root=run_root,
                job_key="nrs_cleanup",
                stage="nrs_cleanup",
                command=(
                    "scripts/py",
                    "-m",
                    "fisheye.cluster.clipped_inference_cleanup",
                    "--plan",
                    str(run_root / "plan.json"),
                    "--cache-root",
                    str(cache_root.expanduser().resolve()),
                    "--package-root",
                    str(package_root.expanduser().resolve()),
                    "--apply",
                    "--output-json",
                    str(cleanup_report),
                ),
                resources=LsfResources(
                    queue="short", ncores=1, mem_gb=4, walltime="1:00"
                ),
                upstream=(registry_key,),
                expected_outputs=(cleanup_report,),
            )
        )
    fragments.append(
        LsfWorkflowFragment(
            fragment_id="campaign_finalize",
            jobs=tuple(jobs[campaign_job_start:]),
            requires=tuple(
                f"validated_analysis:{safe_component(target.target_id, default='target', max_length=56)}"
                for target in targets
            ),
            provides=(
                ("registry_reconciled", "nrs_cache_cleaned")
                if cleanup_nrs_after_success
                else ("registry_reconciled",)
            ),
            metadata={
                "module": "campaign_finalize",
                "target_count": len(targets),
                "cleanup_nrs_after_success": cleanup_nrs_after_success,
            },
        )
    )

    planned_job_keys = [job.job_key for fragment in fragments for job in fragment.jobs]
    duplicate_job_keys = sorted(
        key for key, count in Counter(planned_job_keys).items() if count > 1
    )
    if duplicate_job_keys:
        raise ValueError(
            "Clipped inference fragments contain duplicate jobs: "
            + ", ".join(duplicate_job_keys)
        )
    workflow = compose_lsf_workflow(
        workflow_id=workflow_id,
        family=FAMILY,
        fragments=tuple(fragments),
        metadata={
            "workflow_scope": "full",
            "target_count": len(targets),
            "clip_count": sum(len(target["clips"]) for target in target_payloads),
            "model_bindings_are_exact": True,
            "all_compute_runs_under_lsf": True,
            "resume_existing_detections": resume_existing_detections,
            "scheduler_submission_count": len(jobs),
            "execution_task_count": sum(
                len(job.execution_group.tasks) if job.execution_group else 1
                for job in jobs
            ),
            "array_submission_count": sum(
                1
                for job in jobs
                if job.execution_group is not None
                and job.execution_group.mode is LsfExecutionMode.ARRAY
            ),
            "bundle_submission_count": sum(
                1
                for job in jobs
                if job.execution_group is not None
                and job.execution_group.mode is LsfExecutionMode.BUNDLE
            ),
            "registered_dish_geometry": {
                "gate_requirement": gate_requirement,
                "gate_run": gate_run,
                "selection_policy_id": policy_id,
            },
        },
    )
    return ClippedInferencePlan(
        run_label=label,
        workflow_id=workflow_id,
        workflow_scope=scope,
        repo=repo,
        palette_commit=repo_commit,
        registry=registry_path,
        run_root=run_root,
        targets=tuple(targets),
        target_plans=tuple(target_payloads),
        model_bindings={
            "detection": detection_binding,
            "pose": pose_binding,
            "subject_masks": subject_binding,
        },
        max_active_targets=max_active_targets,
        cleanup_nrs_after_success=cleanup_nrs_after_success,
        resume_existing_detections=resume_existing_detections,
        encoded_mask_packages=encoded_mask_packages,
        subject_mask_publication_profile=subject_mask_publication_profile,
        detect_array_concurrency=int(detect_array_concurrency),
        gpu_array_concurrency=int(gpu_array_concurrency),
        cache_array_concurrency=int(cache_array_concurrency),
        mask_package_array_concurrency=int(mask_package_array_concurrency),
        detect_refine_bundle_concurrency=int(detect_refine_bundle_concurrency),
        registered_gate_requirement=gate_requirement,
        registered_gate_run=gate_run,
        selection_policy_id=policy_id,
        lsf_workflow=workflow,
    )


def materialize_plan_bundle(plan: ClippedInferencePlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"Run root contains a different immutable plan: {plan_path}"
            )
        expected_lsf = plan.lsf_workflow.to_json()
        if (
            not lsf_path.is_file()
            or json.loads(lsf_path.read_text(encoding="utf-8")) != expected_lsf
        ):
            raise FileExistsError(
                f"Run root has mismatched LSF plan evidence: {lsf_path}"
            )
        for target_payload in plan.target_plans:
            detection_plan_path = Path(str(target_payload["detection_plan_path"]))
            if not detection_plan_path.is_file():
                raise FileExistsError(
                    f"Run root is missing immutable detection-plan evidence: {detection_plan_path}"
                )
        return existing
    for name in (
        "logs",
        "status",
        "progress",
        "targets",
        "cache_jobs",
        "ledger",
        "hybrid_crop",
        "receipts",
        "validation",
        "registry",
        "cleanup",
    ):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    for target_payload in plan.target_plans:
        target_safe = safe_component(
            str(target_payload["target_id"]), default="target", max_length=56
        )
        target_dir = plan.run_root / "targets" / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)
        target = next(
            item
            for item in plan.targets
            if item.target_id == target_payload["target_id"]
        )
        if plan.workflow_scope == WORKFLOW_SCOPE_DOWNSTREAM:
            authority = target_payload["native_detection_authority"]
            detection_plan = {
                "schema": "palette.clipped_downstream_layout_plan.v1",
                "workflow_id": str(target_payload["target_label"]),
                "recording_id": target.recording_id,
                "analysis_zarr": str(target.analysis_zarr),
                "external_refined_detect_run": (target.finalized_refined_detect_run),
                "work_units": _clipped_layout_work_units(
                    target,
                    camera_serial=str(authority["camera_serial"]),
                ),
            }
        else:
            detection_binding = plan.model_bindings["detection"]
            detection_plan = _artifact_backed_detection_plan(
                build_detection_plan(
                    target.recording_dir,
                    analysis_zarr=target.analysis_zarr,
                    model=detection_binding.path,
                    config=(
                        plan.repo / "configs" / "fisheye" / "yolo_detect_config.yaml"
                    ),
                    workflow_id=str(target_payload["target_label"]),
                    output_dir=target_dir / "detection_artifacts",
                )
            )
        write_json_snapshot(target_dir / "detection_plan.json", detection_plan)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.lsf_workflow.to_json())
    return payload


def apply_plan(
    plan: ClippedInferencePlan,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    submission = plan.run_root / "lsf_submission.json"
    if submission.exists():
        raise FileExistsError(f"Submission evidence already exists: {submission}")
    materialize_plan_bundle(plan)
    return submit_lsf_workflow(
        plan.lsf_workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=submission,
        runner=runner,
    )


def build_ssh_bsub_runner(submit_host: str) -> CommandRunner:
    """Return a runner that uses the Citrus poller only for individual bsub calls."""

    host = str(submit_host).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host):
        raise ValueError(f"Unsafe LSF submit host: {submit_host!r}")

    def runner(
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        text: bool = True,
        capture_output: bool = True,
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        argv = [str(value) for value in command]
        if not argv or argv[0] != "bsub":
            raise ValueError("The Citrus submission runner accepts only bsub commands.")
        remote_cwd = Path(cwd or DEFAULT_REPO).expanduser()
        remote_command = f"cd {shell_join((remote_cwd,))} && {shell_join(argv)}"
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, remote_command],
            text=text,
            capture_output=capture_output,
        )

    return runner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-scope",
        choices=WORKFLOW_SCOPES,
        default=WORKFLOW_SCOPE_FULL,
        help=(
            "Compose the full analysis DAG (default), stop after canonical "
            "recording-level refined detections, or start downstream analysis "
            "from an exact finalized refined-detection run."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--detection-set-id",
        help="Required for full and detection workflow scopes.",
    )
    parser.add_argument(
        "--detection-run-id",
        help="Required for full and detection workflow scopes.",
    )
    parser.add_argument(
        "--pose-set-id", help="Required for full and downstream workflow scopes."
    )
    parser.add_argument(
        "--pose-run-id", help="Required for full and downstream workflow scopes."
    )
    parser.add_argument(
        "--subject-mask-set-id",
        help="Required for full and downstream workflow scopes.",
    )
    parser.add_argument(
        "--subject-mask-run-id",
        help="Required for full and downstream workflow scopes.",
    )
    parser.add_argument("--subject-mask-coverage-class", default="dense_all_components")
    parser.add_argument(
        "--subject-mask-component-coverage-key", default="body+eyes+swim_bladder"
    )
    parser.add_argument("--subject-mask-label-schema-id", default="subject_v1_union")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument(
        "--submit-host",
        default=os.environ.get("PALETTE_LSF_SUBMIT_HOST", "login1-citrus-poller"),
        help="SSH poller used only for individual bsub commands in --apply mode.",
    )
    parser.add_argument(
        "--cache-bundle-size",
        type=int,
        default=8,
        help=(
            "Clip caches per one-L4 decode bundle (default: 8, selected from the "
            "2026-07-17 L4 concurrency canary)."
        ),
    )
    parser.add_argument("--max-active-targets", type=int, default=3)
    parser.add_argument("--detect-array-concurrency", type=int, default=8)
    parser.add_argument(
        "--gpu-array-concurrency",
        type=int,
        default=4,
        help=(
            "Per-array GPU task limit. Keypoint and subject-mask arrays may run "
            "together, so their combined recording-level maximum is twice this value."
        ),
    )
    parser.add_argument("--cache-array-concurrency", type=int, default=2)
    parser.add_argument("--mask-package-array-concurrency", type=int, default=4)
    parser.add_argument("--detect-refine-bundle-concurrency", type=int, default=4)
    parser.add_argument(
        "--keypoint-gpu-queue",
        choices=("gpu_t4", "gpu_l4"),
        default="gpu_t4",
        help="GPU queue for downstream keypoint clip arrays.",
    )
    parser.add_argument(
        "--subject-mask-gpu-queue",
        choices=("gpu_l4", "gpu_t4"),
        default="gpu_l4",
        help="GPU queue for downstream subject-mask clip arrays.",
    )
    parser.add_argument(
        "--registered-gate-requirement",
        choices=tuple(sorted(REGISTERED_GATE_REQUIREMENTS)),
        default="off",
    )
    parser.add_argument(
        "--registered-gate-run",
        help=(
            "Consume an exact existing gate. Omit with requirement=required to "
            "materialize a target-specific gate from each active geometry selection."
        ),
    )
    parser.add_argument(
        "--selection-policy-id",
        choices=("manual_review_only_v1", "corroborated_acquisition_v1"),
        default="manual_review_only_v1",
    )
    parser.add_argument("--no-cleanup-nrs-after-success", action="store_true")
    parser.add_argument(
        "--resume-existing-detections",
        action="store_true",
        help=(
            "Reuse complete imported detection groups after exact receipt and provenance "
            "validation; all later outputs must still be absent."
        ),
    )
    parser.add_argument(
        "--encoded-mask-packages",
        action="store_true",
        help="Emit v2 globally aligned encoded mask packages and use direct chunk publication.",
    )
    parser.add_argument(
        "--subject-mask-publication-profile",
        choices=SUBJECT_MASK_PUBLICATION_PROFILES,
        default=SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED,
        help=(
            "Receipt-composed is the current default; streaming_rollback_v1 "
            "retains the former serial validation/publication path."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets = load_target_manifest(args.manifest)
    plan = build_plan(
        targets=targets,
        run_label=args.run_label,
        repo=args.repo,
        registry_path=args.registry,
        run_root=args.run_root,
        detection_set_id=args.detection_set_id,
        detection_run_id=args.detection_run_id,
        workflow_scope=args.workflow_scope,
        pose_set_id=args.pose_set_id,
        pose_run_id=args.pose_run_id,
        subject_mask_set_id=args.subject_mask_set_id,
        subject_mask_run_id=args.subject_mask_run_id,
        subject_mask_coverage_class=args.subject_mask_coverage_class,
        subject_mask_component_coverage_key=args.subject_mask_component_coverage_key,
        subject_mask_label_schema_id=args.subject_mask_label_schema_id,
        cache_root=args.cache_root,
        package_root=args.package_root,
        cache_bundle_size=args.cache_bundle_size,
        max_active_targets=args.max_active_targets,
        cleanup_nrs_after_success=not args.no_cleanup_nrs_after_success,
        resume_existing_detections=args.resume_existing_detections,
        encoded_mask_packages=args.encoded_mask_packages,
        subject_mask_publication_profile=args.subject_mask_publication_profile,
        detect_array_concurrency=args.detect_array_concurrency,
        gpu_array_concurrency=args.gpu_array_concurrency,
        cache_array_concurrency=args.cache_array_concurrency,
        mask_package_array_concurrency=args.mask_package_array_concurrency,
        detect_refine_bundle_concurrency=args.detect_refine_bundle_concurrency,
        keypoint_gpu_queue=args.keypoint_gpu_queue,
        subject_mask_gpu_queue=args.subject_mask_gpu_queue,
        registered_gate_requirement=args.registered_gate_requirement,
        registered_gate_run=args.registered_gate_run,
        selection_policy_id=args.selection_policy_id,
    )
    result = (
        apply_plan(plan, runner=build_ssh_bsub_runner(args.submit_host))
        if args.apply
        else materialize_plan_bundle(plan)
    )
    summary = {
        "status": "submitted" if args.apply else "dry_run",
        "workflow_scope": plan.workflow_scope,
        "plan_path": str(plan.run_root / "plan.json"),
        "lsf_plan_path": str(plan.run_root / "lsf_plan.json"),
        "target_count": len(plan.targets),
        "clip_count": sum(len(target["clips"]) for target in plan.target_plans),
        "job_count": len(plan.lsf_workflow.jobs),
        "execution_task_count": int(
            plan.lsf_workflow.metadata["execution_task_count"]
            if plan.lsf_workflow.metadata is not None
            else len(plan.lsf_workflow.jobs)
        ),
        "models": {
            name: binding.to_json() for name, binding in plan.model_bindings.items()
        },
        "result": result if args.apply else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"{summary['status']}: {summary['target_count']} targets, "
            f"{summary['clip_count']} clips, {summary['job_count']} LSF submissions, "
            f"{summary['execution_task_count']} execution tasks"
        )
        print(f"Plan: {summary['plan_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
