"""Plan a complete registry-pinned clipped-recording inference campaign on LSF."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.keypoints.common import (
    resolve_pose_model_binding,
    safe_component,
    validate_registered_analysis_zarr,
)
from fisheye.cluster.lsf import (
    CommandRunner,
    LsfDependency,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    shell_join,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN, build_runtime_command
from fisheye.registry.db import Registry
from fisheye.registry.model_resolution import (
    load_candidates,
    load_subject_mask_model_candidates,
    load_target_profile,
    resolve_recording_id,
    verify_deployment_artifact_content,
)
from fisheye.utils.plan_clipped_detect_refine_workflow import build_plan as build_detection_plan
from fisheye.utils.validate_imported_run_group import validate_imported_run_group


PLAN_SCHEMA = "palette.clipped_inference_bsub_plan.v1"
TARGET_MANIFEST_SCHEMA = "palette.clipped_inference_targets.v1"
FAMILY = "clipped_inference"
DEFAULT_REPO = Path("/groups/johnson/johnsonlab/jeremy/gitrepos/palette")
DEFAULT_REGISTRY = Path("/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite")
DEFAULT_CACHE_ROOT = Path("/nrs/johnson/palette_staging/flat_roi_cache")
DEFAULT_PACKAGE_ROOT = Path("/nrs/johnson/palette_staging/refined_subject_mask_clip_packages")


@dataclass(frozen=True)
class CampaignTarget:
    target_id: str
    recording_id: str
    recording_dir: Path
    analysis_zarr: Path

    def to_json(self) -> dict[str, str]:
        return {
            "target_id": self.target_id,
            "recording_id": self.recording_id,
            "recording_dir": str(self.recording_dir),
            "analysis_zarr": str(self.analysis_zarr),
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
    repo: Path
    registry: Path
    run_root: Path
    targets: tuple[CampaignTarget, ...]
    target_plans: tuple[Mapping[str, Any], ...]
    model_bindings: Mapping[str, ModelBinding]
    max_active_targets: int
    cleanup_nrs_after_success: bool
    resume_existing_detections: bool
    encoded_mask_packages: bool
    lsf_workflow: LsfWorkflow

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "run_label": self.run_label,
            "workflow_id": self.workflow_id,
            "repo": str(self.repo),
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
        if not (recording_dir / "recording_clip_index.json").is_file():
            raise FileNotFoundError(
                f"Target {target_id!r} has no recording_clip_index.json: {recording_dir}"
            )
        if not (analysis_zarr / "zarr.json").is_file():
            raise FileNotFoundError(f"Target {target_id!r} is not a Zarr v3 root: {analysis_zarr}")
        targets.append(
            CampaignTarget(
                target_id=target_id,
                recording_id=recording_id,
                recording_dir=recording_dir,
                analysis_zarr=analysis_zarr,
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
        raise ValueError(f"Registered subject-mask model {run_id!r} has no model_sha256.")
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
    identities = {(item.set_id, item.run_id, item.path, item.sha256) for item in bindings}
    if len(identities) != 1:
        raise ValueError(f"The target cohort does not resolve one common {name} model: {identities}")
    return bindings[0]


def _chain(commands: Sequence[Sequence[str]]) -> tuple[str, ...]:
    return ("bash", "-lc", " && ".join(shell_join(command) for command in commands))


def _job(
    *,
    workflow_id: str,
    repo: Path,
    run_root: Path,
    job_key: str,
    stage: str,
    command: Sequence[str],
    resources: LsfResources,
    upstream: Sequence[str] = (),
    expected_outputs: Sequence[Path] = (),
    cleanup_paths: Sequence[str] = (),
) -> LsfJob:
    safe = safe_component(job_key.replace(":", "_"), default="job", max_length=110)
    runtime_command = build_runtime_command(
        command,
        status_path_template=run_root / "status" / f"{safe}.{RUNTIME_JOB_ID_TOKEN}.json",
        workflow_id=workflow_id,
        family=FAMILY,
        job_key=job_key,
        stage=stage,
        cwd=repo,
        cleanup_path_templates=cleanup_paths,
        expected_output_templates=tuple(str(path) for path in expected_outputs),
    )
    return LsfJob(
        job_key=job_key,
        job_name=safe_component(f"ci_{workflow_id}_{safe}", default=safe, max_length=120),
        command=runtime_command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{safe}.%J.out",
        stderr_path=run_root / "logs" / f"{safe}.%J.err",
        dependency=LsfDependency(tuple(upstream)) if upstream else None,
        metadata={"stage": stage},
    )


def _refuse_output_collisions(
    target_plan: Mapping[str, Any],
    *,
    allow_existing_detections: bool = False,
) -> None:
    zarr = Path(str(target_plan["analysis_zarr"]))
    outputs: list[Path] = []
    for clip in target_plan["clips"]:
        if not allow_existing_detections:
            outputs.append(zarr / str(clip["detect_group_path"]))
        outputs.extend(
            [
                zarr / str(clip["refined_detect_group_path"]),
                zarr / "crop_runs" / str(clip["proxy_crop_run"]),
                zarr / "keypoint_shard_runs" / str(clip["keypoint_shard_run"]),
                zarr / "subject_mask_shard_runs" / str(clip["subject_mask_shard_run"]),
            ]
        )
    outputs.extend(
        [
            zarr / "experiment_index" / "finalized_runs" / str(target_plan["collection_id"]),
            zarr / "crop_runs" / str(target_plan["merged_proxy_crop_run"]),
            zarr / "keypoints_runs" / str(target_plan["keypoint_run"]),
            zarr / "refined_keypoints_runs" / str(target_plan["refined_keypoint_run"]),
            zarr / "refined_subject_masks_runs" / str(target_plan["refined_subject_mask_run"]),
            Path(str(target_plan["cache_dir"])),
            Path(str(target_plan["package_dir"])),
        ]
    )
    collisions = [path for path in outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "Planned immutable outputs already exist: " + ", ".join(str(path) for path in collisions)
        )


def _read_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _validate_existing_detection_for_resume(
    *,
    target: CampaignTarget,
    target_label: str,
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
        raise ValueError(f"Existing detection has no attributes object: {metadata_path}")
    if attrs.get("palette_run_completion_status") != "complete":
        raise ValueError(f"Existing detection is not complete: {metadata_path}")
    provenance = attrs.get("run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Existing detection has no run_provenance: {metadata_path}")
    params = provenance.get("params")
    input_run_ids = provenance.get("input_run_ids")
    clip_context = params.get("clip_context") if isinstance(params, Mapping) else None
    if not isinstance(params, Mapping) or not isinstance(input_run_ids, Mapping):
        raise ValueError(f"Existing detection has incomplete run provenance: {metadata_path}")
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
        "params.clip_context.workflow_id": target_label,
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
        "input_run_ids.model_registry_set_id": input_run_ids.get("model_registry_set_id"),
        "input_run_ids.model_registry_run_id": input_run_ids.get("model_registry_run_id"),
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    artifacts = provenance.get("input_artifacts")
    matching_artifacts = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("role") == "detect_model"
        and artifact.get("path") == str(binding.path)
        and artifact.get("sha256") == binding.sha256
    ] if isinstance(artifacts, list) else []
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
    detection_set_id: str,
    detection_run_id: str,
    pose_set_id: str,
    pose_run_id: str,
    subject_mask_set_id: str,
    subject_mask_run_id: str,
    subject_mask_coverage_class: str = "dense_all_components",
    subject_mask_component_coverage_key: str = "body+eyes+swim_bladder",
    subject_mask_label_schema_id: str = "subject_v1_union",
    cache_root: Path = DEFAULT_CACHE_ROOT,
    package_root: Path = DEFAULT_PACKAGE_ROOT,
    cache_bundle_size: int = 4,
    max_active_targets: int = 3,
    cleanup_nrs_after_success: bool = True,
    resume_existing_detections: bool = False,
    encoded_mask_packages: bool = False,
) -> ClippedInferencePlan:
    if not targets:
        raise ValueError("At least one target is required.")
    if cache_bundle_size <= 0 or max_active_targets <= 0:
        raise ValueError("cache_bundle_size and max_active_targets must be positive.")
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

    detect_bindings: list[ModelBinding] = []
    pose_bindings: list[ModelBinding] = []
    for target in targets:
        validate_registered_analysis_zarr(
            registry_path=registry_path,
            recording_id=target.recording_id,
            analysis_zarr=target.analysis_zarr,
        )
        detect_bindings.append(
            _resolve_ranked_binding(
                registry_path=registry_path,
                target=target,
                task="detect",
                set_id=detection_set_id,
                run_id=detection_run_id,
            )
        )
        pose = resolve_pose_model_binding(
            registry_path=registry_path,
            recording_id=target.recording_id,
            recording_dir=target.recording_dir,
            set_id=pose_set_id,
            run_id=pose_run_id,
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
    detection_binding = _assert_same_binding("detection", detect_bindings)
    pose_binding = _assert_same_binding("pose", pose_bindings)
    subject_binding = _resolve_subject_binding(
        registry_path=registry_path,
        set_id=subject_mask_set_id,
        run_id=subject_mask_run_id,
        coverage_class=subject_mask_coverage_class,
        component_coverage_key=subject_mask_component_coverage_key,
        label_schema_id=subject_mask_label_schema_id,
    )

    gpu = LsfResources(queue="gpu_l4", ncores=8, mem_gb=48, gpus=1, walltime="4:00")
    detect_gpu = LsfResources(queue="gpu_l4", ncores=8, mem_gb=120, gpus=1, walltime="2:00")
    detect_reuse_cpu = LsfResources(queue="short", ncores=1, mem_gb=8, walltime="1:00")
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
    target_payloads: list[dict[str, Any]] = []
    target_validation_keys: list[str] = []

    for target_index, target in enumerate(targets):
        target_safe = safe_component(target.target_id, default=f"target_{target_index}", max_length=56)
        target_label = safe_component(f"{label}_{target_safe}", default=target_safe, max_length=90)
        collection_id = f"refined_detect_collection_{target_label}"
        cache_dir = cache_root.expanduser().resolve() / label / target_safe
        package_dir = package_root.expanduser().resolve() / label / target_safe
        global_mask_grid_manifest = package_dir / "global_mask_chunk_grid.json"
        detection_plan_path = run_root / "targets" / target_safe / "detection_plan.json"
        detection_plan = build_detection_plan(
            target.recording_dir,
            analysis_zarr=target.analysis_zarr,
            model=detection_binding.path,
            config=config_path,
            workflow_id=target_label,
            output_dir=run_root / "targets" / target_safe / "detection_artifacts",
        )
        work_units = list(detection_plan["work_units"])
        clip_ids = [str(unit["clip_id"]) for unit in work_units]
        if len(work_units) != 22:
            raise ValueError(
                f"Target {target.target_id!r} must have exactly 22 clip-camera work units; "
                f"found {len(work_units)}."
            )

        merged_proxy = f"crop_proxy_{target_label}_collection"
        keypoint_run = f"keypoints_registry_{target_label}"
        refined_keypoint_run = f"refined_keypoints_{target_label}"
        refined_subject_mask_run = f"refined_subject_masks_{target_label}"
        clips: list[dict[str, Any]] = []
        for unit in work_units:
            clip = str(unit["clip_id"])
            manifest = cache_dir / f"roi_cache_{target_label}__{clip}.flat_roi_cache.json"
            alias = manifest.with_name(
                f"{manifest.stem}__crop_proxy_{target_label}_{clip}.alias.json"
            )
            clips.append(
                {
                    "clip_id": clip,
                    "clip_index": int(unit["clip_index"]),
                    "camera_serial": str(unit["camera_serial"]),
                    "work_unit_id": str(unit["work_unit_id"]),
                    "video_path": str(unit["source"]["video_path"]),
                    "detect_run": str(unit["run_names"]["detect"]),
                    "detect_group_path": str(unit["zarr_paths"]["detect_target_group_path"]),
                    "quality_run": str(unit["run_names"]["detect_quality"]),
                    "refined_detect_run": str(unit["run_names"]["refined_detect"]),
                    "refined_detect_group_path": str(unit["zarr_paths"]["refined_group_path"]),
                    "cache_manifest": str(manifest),
                    "cache_row_index": str(manifest.with_suffix("").with_suffix(".flat_roi_cache.rows.parquet")),
                    "proxy_crop_run": f"crop_proxy_{target_label}_{clip}",
                    "alias_manifest": str(alias),
                    "keypoint_shard_run": f"keypoint_shard_{target_label}_{clip}",
                    "subject_mask_shard_run": f"subject_mask_shard_{target_label}_{clip}",
                    "refined_mask_package_run": f"refined_subject_masks_{target_label}_{clip}",
                    "package_path": str(package_dir / f"{clip}.tar.gz"),
                }
            )
        target_payload: dict[str, Any] = {
            **target.to_json(),
            "target_label": target_label,
            "detection_plan_path": str(detection_plan_path),
            "collection_id": collection_id,
            "cache_dir": str(cache_dir),
            "package_dir": str(package_dir),
            "global_mask_grid_manifest": str(global_mask_grid_manifest),
            "merged_proxy_crop_run": merged_proxy,
            "keypoint_run": keypoint_run,
            "refined_keypoint_run": refined_keypoint_run,
            "refined_subject_mask_run": refined_subject_mask_run,
            "clips": clips,
        }
        _refuse_output_collisions(
            target_payload,
            allow_existing_detections=resume_existing_detections,
        )
        if resume_existing_detections:
            target_payload["detection_resume_preflight"] = [
                _validate_existing_detection_for_resume(
                    target=target,
                    target_label=target_label,
                    clip=clip,
                    binding=detection_binding,
                )
                for clip in clips
            ]
        target_payloads.append(target_payload)

        gate: tuple[str, ...] = ()
        if target_index >= max_active_targets:
            gate = (target_validation_keys[target_index - max_active_targets],)

        refine_keys: list[str] = []
        for unit, clip in zip(work_units, clips, strict=True):
            clip_id = str(clip["clip_id"])
            detect_key = f"detect:{target_safe}:{clip_id}"
            report = run_root / "targets" / target_safe / "detection_reports" / f"{clip_id}.json"
            detect_command = [
                "scripts/py", "-m", "fisheye.utils.run_clipped_detection_work_unit",
                "--video", str(clip["video_path"]),
                "--target-zarr", str(target.analysis_zarr),
                "--target-group-path", str(clip["detect_group_path"]),
                "--model", str(detection_binding.path),
                "--model-sha256", detection_binding.sha256,
                "--model-registry-set-id", detection_binding.set_id,
                "--model-registry-run-id", detection_binding.run_id,
                "--config", str(config_path),
                "--workflow-id", target_label,
                "--recording-id", target.recording_id,
                "--clip-id", clip_id,
                "--clip-index", str(clip["clip_index"]),
                "--camera-serial", str(clip["camera_serial"]),
                "--recording-frame-index", str(target.recording_dir / "recording_frame_index.parquet"),
                "--run-name", str(clip["detect_run"]),
                "--report", str(report),
                "--batch-size", "16",
                "--decode-backend", "pynvvc_luma_rgb",
            ]
            if resume_existing_detections:
                detect_command.append("--reuse-existing")
            jobs.append(
                _job(
                    workflow_id=workflow_id,
                    repo=repo,
                    run_root=run_root,
                    job_key=detect_key,
                    stage="detect_reuse" if resume_existing_detections else "detect",
                    command=detect_command,
                    resources=detect_reuse_cpu if resume_existing_detections else detect_gpu,
                    upstream=gate,
                    expected_outputs=(target.analysis_zarr / str(clip["detect_group_path"]) / "zarr.json", report),
                    cleanup_paths=(
                        ()
                        if resume_existing_detections
                        else (
                            f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}/palette_clipped_detection",
                        )
                    ),
                )
            )
            refine_key = f"detect_refine:{target_safe}:{clip_id}"
            refine_keys.append(refine_key)
            quality = [
                "scripts/py", "-m", "fisheye.refinement.detect_quality",
                str(target.analysis_zarr), "--run", str(clip["detect_run"]),
                "--detect-family-path", str(Path(str(clip["detect_group_path"])).parent),
                "--threshold", "100.0", "--threshold-mode", "scaled",
                "--threshold-reference-width", "640.0", "--quality-run-name", str(clip["quality_run"]),
            ]
            refine = [
                "scripts/py", "-m", "fisheye.refinement.refine_detect",
                str(target.analysis_zarr), "--detect-run", str(clip["detect_run"]),
                "--detect-family-path", str(Path(str(clip["detect_group_path"])).parent),
                "--refined-family-path", str(Path(str(clip["refined_detect_group_path"])).parent),
                "--quality-run", str(clip["quality_run"]), "--config", str(refine_config),
                "--run-name", str(clip["refined_detect_run"]), "--per-frame-top-k", "1",
            ]
            validate = [
                "scripts/py", "-m", "fisheye.utils.validate_refined_detect_run",
                str(target.analysis_zarr), "--target-group-path", str(clip["refined_detect_group_path"]),
            ]
            jobs.append(
                _job(
                    workflow_id=workflow_id, repo=repo, run_root=run_root,
                    job_key=refine_key, stage="detect_refine",
                    command=_chain((quality, refine, validate)), resources=cpu,
                    upstream=(detect_key,),
                    expected_outputs=(target.analysis_zarr / str(clip["refined_detect_group_path"]) / "zarr.json",),
                )
            )

        collection_key = f"detect_collection:{target_safe}"
        collection_report = run_root / "targets" / target_safe / "detection_collection.json"
        collection_command = [
            "scripts/py", "-m", "fisheye.utils.finalize_clipped_detect_refine_workflow",
            str(detection_plan_path), "--collection-id", collection_id,
            "--no-require-stage-status", "--apply", "--output-json", str(collection_report),
        ]
        jobs.append(
            _job(
                workflow_id=workflow_id, repo=repo, run_root=run_root,
                job_key=collection_key, stage="detect_collection", command=collection_command,
                resources=cpu, upstream=tuple(refine_keys),
                expected_outputs=(target.analysis_zarr / "experiment_index" / "finalized_runs" / collection_id / "zarr.json", collection_report),
            )
        )

        cache_keys: list[str] = []
        for bundle_index, start in enumerate(range(0, len(clips), cache_bundle_size)):
            bundle_clips = clips[start : start + cache_bundle_size]
            cache_key = f"cache:{target_safe}:{bundle_index:02d}"
            cache_keys.append(cache_key)
            cache_command = [
                "bash", "scripts/submit_clipped_collection_flat_roi_cache_bundle_bsub.sh",
                "--zarr", str(target.analysis_zarr), "--collection-id", collection_id,
                "--recording-frame-index", str(target.recording_dir / "recording_frame_index.parquet"),
                "--public-cache-dir", str(cache_dir), "--run-id", f"{target_label}_{bundle_index:02d}",
                "--run-label", f"roi_cache_{target_label}", "--log-dir", str(run_root / "cache_jobs" / target_safe),
                "--max-workers", str(len(bundle_clips)), "--gpus", "0", "--run-direct",
            ]
            for clip in bundle_clips:
                cache_command.extend(["--clip-id", str(clip["clip_id"])])
            jobs.append(
                _job(
                    workflow_id=workflow_id, repo=repo, run_root=run_root,
                    job_key=cache_key, stage="roi_cache", command=cache_command,
                    resources=cache_gpu, upstream=(collection_key,),
                    expected_outputs=tuple(Path(str(clip["cache_manifest"])) for clip in bundle_clips),
                )
            )

        proxy_key = f"proxy:{target_safe}"
        proxy_commands: list[list[str]] = []
        for clip in clips:
            proxy_commands.append(
                [
                    "scripts/py", "-m", "fisheye.utils.create_clipped_collection_proxy_crop_run",
                    str(target.analysis_zarr), str(clip["cache_manifest"]),
                    "--proxy-run", str(clip["proxy_crop_run"]),
                    "--alias-manifest", str(clip["alias_manifest"]), "--json",
                ]
            )
        jobs.append(
            _job(
                workflow_id=workflow_id, repo=repo, run_root=run_root,
                job_key=proxy_key, stage="proxy_crop", command=_chain(proxy_commands),
                resources=cpu, upstream=tuple(cache_keys),
                expected_outputs=tuple(Path(str(clip["alias_manifest"])) for clip in clips),
            )
        )

        keypoint_keys: list[str] = []
        mask_keys: list[str] = []
        for clip in clips:
            clip_id = str(clip["clip_id"])
            keypoint_key = f"keypoints:{target_safe}:{clip_id}"
            keypoint_keys.append(keypoint_key)
            keypoint_command = [
                "scripts/py", "-m", "fisheye.utils.run_keypoints_with_registry_model",
                "--recording-dir", str(target.recording_dir), "--output", str(target.analysis_zarr),
                "--registry", str(registry_path), "--set-id", pose_binding.set_id,
                "--model-run-id", pose_binding.run_id, "--require-unique",
                "--run-name", str(clip["keypoint_shard_run"]), "--output-parent", "keypoint_shard_runs",
                "--crop-run", str(clip["proxy_crop_run"]), "--pose-schema", "traditional_v2",
                "--batch-size", "256", "--device", "0", "--roi-cache-manifest", str(clip["alias_manifest"]),
                "--stage-roi-cache-to-scratch", "--keypoint-roi-shard-rows", "131072",
                "--keypoint-frame-shard-rows", "131072",
                "--progress-jsonl", str(run_root / "progress" / f"keypoints_{target_safe}_{clip_id}.jsonl"),
            ]
            jobs.append(
                _job(
                    workflow_id=workflow_id, repo=repo, run_root=run_root,
                    job_key=keypoint_key, stage="keypoints", command=keypoint_command,
                    resources=gpu, upstream=(proxy_key,),
                    expected_outputs=(target.analysis_zarr / "keypoint_shard_runs" / str(clip["keypoint_shard_run"]) / "zarr.json",),
                )
            )

            mask_key = f"subject_masks:{target_safe}:{clip_id}"
            mask_keys.append(mask_key)
            mask_command = [
                "scripts/py", "-m", "fisheye.segmentation.infer_unet_subject_masks",
                str(target.analysis_zarr), "--resolve-model-from-registry", "--registry", str(registry_path),
                "--model-set-id", subject_binding.set_id, "--model-run-id", subject_binding.run_id,
                "--model-coverage-class", subject_mask_coverage_class,
                "--model-component-coverage-key", subject_mask_component_coverage_key,
                "--model-label-schema-id", subject_mask_label_schema_id, "--model-require-unique",
                "--run-name", str(clip["subject_mask_shard_run"]), "--output-parent", "subject_mask_shard_runs",
                "--crop-run", str(clip["proxy_crop_run"]), "--source-collection-id", collection_id,
                "--source-collection-path", f"experiment_index/finalized_runs/{collection_id}",
                "--source-clip-id", clip_id, "--source-clip-index", str(clip["clip_index"]),
                "--source-work-unit-id", str(clip["work_unit_id"]),
                "--source-roi-cache-alias-manifest", str(clip["alias_manifest"]),
                "--source-roi-cache-row-index-path", str(clip["cache_row_index"]),
                "--roi-cache-manifest", str(clip["alias_manifest"]), "--roi-cache-policy", "never",
                "--batch-size", "128", "--device", "0", "--mask-probs-dtype", "uint8",
                "--mask-probs-chunk-rois", "32", "--mask-probs-shard-rois", "2048",
                "--no-write-masks-roi", "--async-output", "--output-queue-size", "2",
                "--no-progress", "--defer-registry-status",
            ]
            jobs.append(
                _job(
                    workflow_id=workflow_id, repo=repo, run_root=run_root,
                    job_key=mask_key, stage="subject_mask_inference", command=mask_command,
                    resources=gpu, upstream=(proxy_key,),
                    expected_outputs=(target.analysis_zarr / "subject_mask_shard_runs" / str(clip["subject_mask_shard_run"]) / "zarr.json",),
                )
            )

        keypoint_finalize_key = f"keypoint_finalize:{target_safe}"
        merge_proxy = [
            "scripts/py", "-m", "fisheye.utils.merge_clipped_proxy_crop_runs",
            str(target.analysis_zarr), "--output-run", merged_proxy, "--json",
        ]
        for clip in clips:
            merge_proxy.extend(["--source-crop-run", str(clip["proxy_crop_run"])])
        finalize_keypoints = [
            "scripts/py", "-m", "fisheye.utils.finalize_keypoint_shards",
            str(target.analysis_zarr), "--target-crop-run", merged_proxy,
            "--output-run", keypoint_run, "--json",
        ]
        for clip in clips:
            finalize_keypoints.extend(["--shard-run", str(clip["keypoint_shard_run"])])
        jobs.append(
            _job(
                workflow_id=workflow_id, repo=repo, run_root=run_root,
                job_key=keypoint_finalize_key, stage="keypoint_finalize",
                command=_chain((merge_proxy, finalize_keypoints)), resources=final_cpu,
                upstream=tuple(keypoint_keys),
                expected_outputs=(
                    target.analysis_zarr / "crop_runs" / merged_proxy / "zarr.json",
                    target.analysis_zarr / "keypoints_runs" / keypoint_run / "zarr.json",
                ),
            )
        )
        keypoint_refine_key = f"keypoint_refine:{target_safe}"
        keypoint_refine = [
            "scripts/py", "-m", "fisheye.refinement.refine_keypoints",
            str(target.analysis_zarr), "--keypoint-run", keypoint_run,
            "--run-name", refined_keypoint_run, "--chunk-size", "2048",
            "--scheduler", "threads", "--num-workers", "4", "--no-post-audit",
        ]
        jobs.append(
            _job(
                workflow_id=workflow_id, repo=repo, run_root=run_root,
                job_key=keypoint_refine_key, stage="keypoint_refine", command=keypoint_refine,
                resources=cpu, upstream=(keypoint_finalize_key,),
                expected_outputs=(target.analysis_zarr / "refined_keypoints_runs" / refined_keypoint_run / "zarr.json",),
            )
        )

        mask_grid_key: str | None = None
        if encoded_mask_packages:
            mask_grid_key = f"mask_grid:{target_safe}"
            mask_grid_command = [
                "scripts/py", "-m", "fisheye.utils.prepare_refined_subject_mask_chunk_grid",
                "--zarr", str(target.analysis_zarr),
                "--crop-run", merged_proxy,
                "--output-manifest", str(global_mask_grid_manifest),
                "--mask-label", "subject_body",
                "--mask-label", "eye_left",
                "--mask-label", "eye_right",
                "--mask-label", "swim_bladder",
                "--mask-height", "512", "--mask-width", "512",
                "--dense-mask-row-chunk", "128", "--json",
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

        package_keys: list[str] = []
        for clip, mask_key in zip(clips, mask_keys, strict=True):
            clip_id = str(clip["clip_id"])
            package_key = f"mask_package:{target_safe}:{clip_id}"
            package_keys.append(package_key)
            package_command = [
                "scripts/py", "-m", "fisheye.utils.finalize_subject_mask_clip_package",
                "--source-zarr", str(target.analysis_zarr),
                "--subject-shard-run", str(clip["subject_mask_shard_run"]),
                "--target-crop-run", merged_proxy,
                "--refined-run", str(clip["refined_mask_package_run"]),
                "--package-path", str(clip["package_path"]),
                "--component", "subject_body", "--component", "eyes_union", "--component", "swim_bladder",
                "--chunk-size", "256", "--metric-level", "cheap",
                "--mask-storage", "dense_and_bitpacked", "--dense-mask-row-chunk", "128",
                "--execution-backend", "process_shards", "--num-workers", "8",
                "--postcompute-backend", "process_shards", "--postcompute-num-workers", "8",
                "--postcompute-chunk-size", "256",
                "--assignment-keypoint-group", "refined_keypoints_runs",
                "--assignment-keypoints-run", refined_keypoint_run,
                "--no-write-component-contours", "--json",
            ]
            if encoded_mask_packages:
                package_command.extend(
                    [
                        "--global-mask-grid-manifest", str(global_mask_grid_manifest),
                        "--encoded-mask-copy-workers", "8",
                    ]
                )
            package_upstream = [mask_key, keypoint_refine_key]
            if mask_grid_key is not None:
                package_upstream.append(mask_grid_key)
            jobs.append(
                _job(
                    workflow_id=workflow_id, repo=repo, run_root=run_root,
                    job_key=package_key, stage="subject_mask_refine_package",
                    command=package_command, resources=final_cpu,
                    upstream=tuple(package_upstream),
                    expected_outputs=(Path(str(clip["package_path"])),),
                )
            )

        mask_import_key = f"mask_import:{target_safe}"
        mask_import = [
            "scripts/py", "-m", "fisheye.utils.import_refined_subject_mask_clip_packages",
            "--zarr", str(target.analysis_zarr), "--output-run", refined_subject_mask_run,
            "--expected-target-crop-run", merged_proxy, "--array-copy-workers", "8",
            "--encoded-copy-workers", "32", "--json",
        ]
        for clip in clips:
            mask_import.extend(["--package", str(clip["package_path"])])
        jobs.append(
            _job(
                workflow_id=workflow_id, repo=repo, run_root=run_root,
                job_key=mask_import_key, stage="subject_mask_collection_import",
                command=mask_import, resources=import_cpu, upstream=tuple(package_keys),
                expected_outputs=(target.analysis_zarr / "refined_subject_masks_runs" / refined_subject_mask_run / "zarr.json",),
            )
        )

        validation_key = f"validate:{target_safe}"
        validation_report = run_root / "validation" / f"{target_safe}.json"
        validation_command = [
            "scripts/py", "-m", "fisheye.cluster.clipped_inference_validate",
            "--plan", str(run_root / "plan.json"), "--target-id", target.target_id,
            "--output-json", str(validation_report),
        ]
        jobs.append(
            _job(
                workflow_id=workflow_id, repo=repo, run_root=run_root,
                job_key=validation_key, stage="validation", command=validation_command,
                resources=final_cpu, upstream=(mask_import_key,), expected_outputs=(validation_report,),
            )
        )
        target_validation_keys.append(validation_key)

    registry_key = "registry_finalize"
    registry_report = run_root / "registry" / "reconcile.json"
    registry_command = [
        "scripts/py", "-m", "fisheye.cluster.clipped_inference_registry_finalize",
        "--plan", str(run_root / "plan.json"), "--output-json", str(registry_report),
    ]
    jobs.append(
        _job(
            workflow_id=workflow_id, repo=repo, run_root=run_root,
            job_key=registry_key, stage="registry_finalize", command=registry_command,
            resources=cpu, upstream=tuple(target_validation_keys), expected_outputs=(registry_report,),
        )
    )
    if cleanup_nrs_after_success:
        cleanup_report = run_root / "cleanup" / "nrs_cleanup.json"
        jobs.append(
            _job(
                workflow_id=workflow_id, repo=repo, run_root=run_root,
                job_key="nrs_cleanup", stage="nrs_cleanup",
                command=(
                    "scripts/py", "-m", "fisheye.cluster.clipped_inference_cleanup",
                    "--plan", str(run_root / "plan.json"),
                    "--cache-root", str(cache_root.expanduser().resolve()),
                    "--package-root", str(package_root.expanduser().resolve()),
                    "--apply", "--output-json", str(cleanup_report),
                ),
                resources=LsfResources(queue="short", ncores=1, mem_gb=4, walltime="1:00"),
                upstream=(registry_key,), expected_outputs=(cleanup_report,),
            )
        )

    workflow = LsfWorkflow(
        workflow_id=workflow_id,
        family=FAMILY,
        jobs=tuple(jobs),
        metadata={
            "target_count": len(targets),
            "clip_count": sum(len(target["clips"]) for target in target_payloads),
            "model_bindings_are_exact": True,
            "all_compute_runs_under_lsf": True,
            "resume_existing_detections": resume_existing_detections,
        },
    )
    return ClippedInferencePlan(
        run_label=label,
        workflow_id=workflow_id,
        repo=repo,
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
        lsf_workflow=workflow,
    )


def materialize_plan_bundle(plan: ClippedInferencePlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Run root contains a different immutable plan: {plan_path}")
        expected_lsf = plan.lsf_workflow.to_json()
        if not lsf_path.is_file() or json.loads(lsf_path.read_text(encoding="utf-8")) != expected_lsf:
            raise FileExistsError(f"Run root has mismatched LSF plan evidence: {lsf_path}")
        for target_payload in plan.target_plans:
            detection_plan_path = Path(str(target_payload["detection_plan_path"]))
            if not detection_plan_path.is_file():
                raise FileExistsError(
                    f"Run root is missing immutable detection-plan evidence: {detection_plan_path}"
                )
        return existing
    for name in ("logs", "status", "progress", "targets", "cache_jobs", "validation", "registry", "cleanup"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    for target_payload in plan.target_plans:
        target_safe = safe_component(str(target_payload["target_id"]), default="target", max_length=56)
        target_dir = plan.run_root / "targets" / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)
        target = next(item for item in plan.targets if item.target_id == target_payload["target_id"])
        detection_binding = plan.model_bindings["detection"]
        detection_plan = build_detection_plan(
            target.recording_dir,
            analysis_zarr=target.analysis_zarr,
            model=detection_binding.path,
            config=plan.repo / "configs" / "fisheye" / "yolo_detect_config.yaml",
            workflow_id=str(target_payload["target_label"]),
            output_dir=target_dir / "detection_artifacts",
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
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--detection-set-id", required=True)
    parser.add_argument("--detection-run-id", required=True)
    parser.add_argument("--pose-set-id", required=True)
    parser.add_argument("--pose-run-id", required=True)
    parser.add_argument("--subject-mask-set-id", required=True)
    parser.add_argument("--subject-mask-run-id", required=True)
    parser.add_argument("--subject-mask-coverage-class", default="dense_all_components")
    parser.add_argument("--subject-mask-component-coverage-key", default="body+eyes+swim_bladder")
    parser.add_argument("--subject-mask-label-schema-id", default="subject_v1_union")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument(
        "--submit-host",
        default=os.environ.get("PALETTE_LSF_SUBMIT_HOST", "login1-citrus-poller"),
        help="SSH poller used only for individual bsub commands in --apply mode.",
    )
    parser.add_argument("--cache-bundle-size", type=int, default=4)
    parser.add_argument("--max-active-targets", type=int, default=3)
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
    )
    result = (
        apply_plan(plan, runner=build_ssh_bsub_runner(args.submit_host))
        if args.apply
        else materialize_plan_bundle(plan)
    )
    summary = {
        "status": "submitted" if args.apply else "dry_run",
        "plan_path": str(plan.run_root / "plan.json"),
        "lsf_plan_path": str(plan.run_root / "lsf_plan.json"),
        "target_count": len(plan.targets),
        "clip_count": sum(len(target["clips"]) for target in plan.target_plans),
        "job_count": len(plan.lsf_workflow.jobs),
        "models": {name: binding.to_json() for name, binding in plan.model_bindings.items()},
        "result": result if args.apply else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"{summary['status']}: {summary['target_count']} targets, "
            f"{summary['clip_count']} clips, {summary['job_count']} LSF jobs"
        )
        print(f"Plan: {summary['plan_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
