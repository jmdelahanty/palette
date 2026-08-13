"""Plan explicit whole-recording keypoint prediction and refinement LSF jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.keypoints.common import (
    DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
    FlatRoiCacheBinding,
    KeypointInputCapability,
    KeypointRunNames,
    PoseModelBinding,
    build_keypoint_run_names,
    build_prediction_job,
    build_refinement_job,
    resolve_pose_model_binding,
    resolve_keypoint_v2_publication_storage,
    safe_component,
    validate_flat_roi_cache_binding,
    validate_keypoint_input_dag,
    validate_registered_geometry_crop_authority,
    validate_registered_analysis_zarr,
)
from fisheye.cluster.lsf import (
    CommandRunner,
    LsfDependency,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    build_bsub_command,
    shell_join,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN, build_runtime_command
from fisheye.shared.model_input_transform import (
    ModelInputTransform,
)
from fisheye.shared.pose_model_input_contract import (
    PoseModelInputContractBinding,
    PoseModelInputRuntimePlan,
    load_pose_model_input_contract,
)


TARGET_MANIFEST_SCHEMA = "palette.whole_recording_keypoint_targets.v1"
PLAN_SCHEMA = "palette.whole_recording_keypoint_bsub_plan.v5"
NORMALIZED_TARGETS_SCHEMA = "palette.whole_recording_keypoint_targets.normalized.v1"
DEFAULT_GROUPS_REPO = Path("/groups/johnson/johnsonlab/jeremy/gitrepos/palette")
DEFAULT_REGISTRY = Path(
    "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
)


def _repo_head_commit(repo: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "--verify", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


@dataclass(frozen=True)
class WholeRecordingTarget:
    target_id: str
    recording_id: str
    recording_dir: Path
    analysis_zarr: Path
    roi_cache_manifest: Path
    crop_run: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "recording_dir": str(self.recording_dir),
            "analysis_zarr": str(self.analysis_zarr),
            "roi_cache_manifest": str(self.roi_cache_manifest),
        }


@dataclass(frozen=True)
class PlannedWholeRecordingTarget:
    target: WholeRecordingTarget
    model: PoseModelBinding
    cache: FlatRoiCacheBinding
    model_input_contract: PoseModelInputContractBinding
    model_input_runtime: PoseModelInputRuntimePlan
    model_input_transform: ModelInputTransform
    model_input_stride: int
    input_capability: KeypointInputCapability
    run_names: KeypointRunNames
    expected_keypoint_group: Path
    expected_keypoint_quality_group: Path
    expected_refined_keypoint_group: Path
    expected_body_frame_group: Path
    finalization_result: Path
    prediction_job_key: str
    refinement_job_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target.to_json(),
            "model": self.model.to_json(),
            "cache": self.cache.to_json(),
            "model_input_contract": self.model_input_contract.to_json(),
            "model_input_runtime": self.model_input_runtime.to_json(),
            "model_input_transform": self.model_input_transform.to_attrs(),
            "model_input_stride": self.model_input_stride,
            "input_capability": self.input_capability.to_json(),
            "run_names": self.run_names.to_json(),
            "expected_keypoint_group": str(self.expected_keypoint_group),
            "expected_keypoint_quality_group": str(
                self.expected_keypoint_quality_group
            ),
            "expected_refined_keypoint_group": str(
                self.expected_refined_keypoint_group
            ),
            "expected_body_frame_group": str(self.expected_body_frame_group),
            "finalization_result": str(self.finalization_result),
            "prediction_job_key": self.prediction_job_key,
            "refinement_job_key": self.refinement_job_key,
        }


@dataclass(frozen=True)
class WholeRecordingWorkflowPlan:
    manifest_path: Path
    manifest_sha256: str
    run_label: str
    repo: Path
    palette_commit: str
    registry: Path
    run_root: Path
    model_set_id: str
    model_run_id: str
    model_input_contract: PoseModelInputContractBinding
    pose_schema: str
    min_roi_size: int
    model_input_stride: int
    keypoint_storage: dict[str, Any]
    finalization_execution: dict[str, Any]
    registered_gate_requirement: str
    registered_gate_run: str | None
    targets: tuple[PlannedWholeRecordingTarget, ...]
    registry_finalizer_job_key: str
    lsf_workflow: LsfWorkflow

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "run_label": self.run_label,
            "repo": str(self.repo),
            "palette_commit": self.palette_commit,
            "registry": str(self.registry),
            "run_root": str(self.run_root),
            "model_set_id": self.model_set_id,
            "model_run_id": self.model_run_id,
            "model_input_contract": self.model_input_contract.to_json(),
            "pose_schema": self.pose_schema,
            "min_roi_size": self.min_roi_size,
            "model_input_stride": self.model_input_stride,
            "keypoint_storage": self.keypoint_storage,
            "finalization_execution": self.finalization_execution,
            "registered_dish_geometry": {
                "gate_requirement": self.registered_gate_requirement,
                "gate_run": self.registered_gate_run,
            },
            "target_count": len(self.targets),
            "targets": [target.to_json() for target in self.targets],
            "registry_finalizer_job_key": self.registry_finalizer_job_key,
            "lsf_workflow": self.lsf_workflow.to_json(),
        }


def _resolve_manifest_path(value: object, *, base_dir: Path, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Target field {field!r} is required.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_target_manifest(path: Path) -> tuple[WholeRecordingTarget, ...]:
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Target manifest must be a JSON object: {manifest_path}")
    if payload.get("schema") != TARGET_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported target manifest schema {payload.get('schema')!r}; "
            f"expected {TARGET_MANIFEST_SCHEMA!r}."
        )
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Target manifest requires a non-empty 'targets' array.")

    targets: list[WholeRecordingTarget] = []
    for index, raw in enumerate(raw_targets, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Target {index} must be a JSON object.")
        recording_id = str(raw.get("recording_id") or "").strip()
        if not recording_id:
            raise ValueError(f"Target {index} is missing recording_id.")
        target_id = str(raw.get("target_id") or recording_id).strip()
        if not target_id:
            raise ValueError(f"Target {index} has an empty target_id.")
        crop_run_value = str(raw.get("crop_run") or "").strip()
        targets.append(
            WholeRecordingTarget(
                target_id=target_id,
                recording_id=recording_id,
                recording_dir=_resolve_manifest_path(
                    raw.get("recording_dir"),
                    base_dir=manifest_path.parent,
                    field="recording_dir",
                ),
                analysis_zarr=_resolve_manifest_path(
                    raw.get("analysis_zarr"),
                    base_dir=manifest_path.parent,
                    field="analysis_zarr",
                ),
                roi_cache_manifest=_resolve_manifest_path(
                    raw.get("roi_cache_manifest"),
                    base_dir=manifest_path.parent,
                    field="roi_cache_manifest",
                ),
                crop_run=crop_run_value or None,
            )
        )

    expected_count = payload.get("expected_target_count")
    if expected_count is not None and int(expected_count) != len(targets):
        raise ValueError(
            f"Target manifest expected_target_count is {expected_count}, but contains "
            f"{len(targets)} targets."
        )
    target_ids = [target.target_id for target in targets]
    recording_ids = [target.recording_id for target in targets]
    zarr_paths = [target.analysis_zarr for target in targets]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Target manifest target_id values must be unique.")
    if len(set(recording_ids)) != len(recording_ids):
        raise ValueError("Target manifest recording_id values must be unique.")
    if len(set(zarr_paths)) != len(zarr_paths):
        raise ValueError("Target manifest analysis_zarr values must be unique.")
    return tuple(targets)


def _require_target_paths(
    target: WholeRecordingTarget,
    *,
    require_cache_manifest: bool = True,
) -> None:
    if not target.recording_dir.is_dir():
        raise FileNotFoundError(
            f"Recording directory not found for {target.target_id!r}: "
            f"{target.recording_dir}"
        )
    if not target.analysis_zarr.is_dir():
        raise FileNotFoundError(
            f"Analysis Zarr not found for {target.target_id!r}: "
            f"{target.analysis_zarr}"
        )
    try:
        target.analysis_zarr.relative_to(target.recording_dir)
    except ValueError as exc:
        raise ValueError(
            f"Analysis Zarr for {target.target_id!r} is outside its recording "
            f"directory: {target.analysis_zarr}"
        ) from exc
    if require_cache_manifest and not target.roi_cache_manifest.is_file():
        raise FileNotFoundError(
            f"ROI cache manifest not found for {target.target_id!r}: "
            f"{target.roi_cache_manifest}"
        )


def _refuse_output_collisions(
    *,
    target: WholeRecordingTarget,
    run_names: KeypointRunNames,
) -> tuple[Path, Path, Path, Path]:
    keypoint_group = target.analysis_zarr / "keypoints_runs" / run_names.keypoint_run
    refined_group = (
        target.analysis_zarr
        / "refined_keypoints_runs"
        / run_names.refined_keypoint_run
    )
    quality_group = (
        target.analysis_zarr
        / "keypoint_quality_runs"
        / run_names.keypoint_quality_run
    )
    body_frame_group = (
        target.analysis_zarr
        / "analysis"
        / "body_frame_runs"
        / run_names.body_frame_run
    )
    collisions = [
        path
        for path in (keypoint_group, quality_group, refined_group, body_frame_group)
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            f"Planned output already exists for {target.target_id!r}: "
            + ", ".join(str(path) for path in collisions)
        )
    return keypoint_group, quality_group, refined_group, body_frame_group


def build_plan(
    *,
    manifest_path: Path,
    run_label: str,
    repo: Path,
    palette_commit: str,
    registry: Path,
    run_root: Path,
    model_set_id: str,
    model_run_id: str,
    model_input_contract_path: Path,
    pose_schema: str,
    min_roi_size: int,
    batch_size: int,
    device: str,
    input_mode: str,
    model_input_stride: int | None,
    progress_every_batches: int,
    keypoint_roi_shard_rows: int | None = DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    keypoint_frame_shard_rows: int = DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    prediction_resources: LsfResources,
    refinement_resources: LsfResources,
    refine_chunk_size: int,
    refine_scheduler: str,
    refine_num_workers: int,
    refine_memory_limit: str | None,
    finalizer_resources: LsfResources,
    cache_bindings: Mapping[str, FlatRoiCacheBinding] | None = None,
    upstream_jobs: Sequence[LsfJob] = (),
    registered_gate_requirement: str = "off",
    registered_gate_run: str | None = None,
) -> WholeRecordingWorkflowPlan:
    resolved_manifest = manifest_path.expanduser().resolve()
    resolved_repo = repo.expanduser().resolve()
    resolved_registry = registry.expanduser().resolve()
    resolved_run_root = run_root.expanduser().resolve()
    resolved_model_input_contract = model_input_contract_path.expanduser().resolve()
    resolved_run_label = safe_component(run_label, default="keypoints")
    normalized_palette_commit = str(palette_commit).strip().lower()
    if (
        len(normalized_palette_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in normalized_palette_commit
        )
    ):
        raise ValueError("palette_commit must be one full 40-character Git commit.")
    if not resolved_repo.is_dir() or not (resolved_repo / "scripts" / "py").is_file():
        raise FileNotFoundError(
            f"Palette repository or scripts/py not found: {resolved_repo}"
        )
    if _repo_head_commit(resolved_repo) != normalized_palette_commit:
        raise ValueError(
            "palette_commit does not match the exact HEAD at palette_repo."
        )
    if not resolved_registry.is_file():
        raise FileNotFoundError(f"Palette registry not found: {resolved_registry}")
    if not str(model_set_id).strip() or not str(model_run_id).strip():
        raise ValueError("Both model_set_id and model_run_id are required.")
    if int(batch_size) <= 0 or int(progress_every_batches) <= 0:
        raise ValueError("Batch size and progress interval must be positive.")
    if int(min_roi_size) <= 0:
        raise ValueError("Minimum zebrafish ROI size must be positive.")
    if not resolved_model_input_contract.is_file():
        raise FileNotFoundError(
            f"Pose model-input contract not found: {resolved_model_input_contract}"
        )
    if input_mode not in {"model-contract", "numpy-list", "tensor", "auto"}:
        raise ValueError("Unsupported model input-mode assertion.")
    if model_input_stride is not None and (
        type(model_input_stride) is not int or model_input_stride <= 0
    ):
        raise ValueError("Model input stride assertion must be a positive integer.")
    if int(refine_chunk_size) <= 0 or int(refine_num_workers) <= 0:
        raise ValueError("Refinement chunk size and worker count must be positive.")
    gate_requirement = str(registered_gate_requirement).strip()
    if gate_requirement not in {"off", "if_available", "required"}:
        raise ValueError(
            "registered_gate_requirement must be off, if_available, or required."
        )
    gate_run = str(registered_gate_run or "").strip() or None
    if gate_requirement == "required" and gate_run is None:
        raise ValueError("Required registered geometry needs one exact gate run.")
    keypoint_storage = resolve_keypoint_v2_publication_storage(
        legacy_roi_shard_rows=keypoint_roi_shard_rows,
        legacy_frame_shard_rows=keypoint_frame_shard_rows,
    )
    finalization_execution = {
        "requested_legacy_controls": {
            "chunk_size": int(refine_chunk_size),
            "scheduler": str(refine_scheduler),
            "num_workers": int(refine_num_workers),
            "memory_limit": refine_memory_limit,
            "effect_on_v2_finalization": "none",
        },
        "effective": {
            "algorithm": "strict_keypoint_v2_chain_finalizer_v1",
            "write_ownership": "serial_whole_physical_units",
            "storage_planning": "shared_byte_planner_per_array",
            "publication": "atomic_per_run_then_single_root_consolidation",
            "selector_activation": False,
        },
    }

    manifest_bytes = resolved_manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    targets = load_target_manifest(resolved_manifest)
    run_names = build_keypoint_run_names(resolved_run_label)
    jobs: list[LsfJob] = list(upstream_jobs)
    planned_targets: list[PlannedWholeRecordingTarget] = []
    model_identity: tuple[str, str, str, str] | None = None
    contract_identity: tuple[str, str] | None = None
    selected_contract: PoseModelInputContractBinding | None = None

    for target in targets:
        supplied_cache = (
            cache_bindings.get(target.target_id)
            if cache_bindings is not None
            else None
        )
        _require_target_paths(
            target,
            require_cache_manifest=supplied_cache is None,
        )
        validate_registered_analysis_zarr(
            registry_path=resolved_registry,
            recording_id=target.recording_id,
            analysis_zarr=target.analysis_zarr,
        )
        model = resolve_pose_model_binding(
            registry_path=resolved_registry,
            recording_id=target.recording_id,
            recording_dir=target.recording_dir,
            set_id=str(model_set_id),
            run_id=str(model_run_id),
        )
        current_model_identity = (
            model.set_id,
            model.run_id,
            str(model.model_path),
            model.model_sha256,
        )
        if model_identity is None:
            model_identity = current_model_identity
        elif current_model_identity != model_identity:
            raise ValueError(
                "Exact model set/run resolved to inconsistent deployment artifacts "
                f"across targets; {target.target_id!r} resolved {current_model_identity}, "
                f"expected {model_identity}."
            )
        model_contract = load_pose_model_input_contract(
            resolved_model_input_contract,
            model_path=model.model_path,
            expected_set_id=model.set_id,
            expected_run_id=model.run_id,
            expected_model_sha256=model.model_sha256,
        )
        current_contract_identity = (
            model_contract.sha256,
            model_contract.payload_digest,
        )
        if contract_identity is None:
            contract_identity = current_contract_identity
            selected_contract = model_contract
        elif current_contract_identity != contract_identity:
            raise ValueError(
                "Pose model-input contract resolved inconsistently across targets."
            )
        if supplied_cache is None:
            cache = validate_flat_roi_cache_binding(
                manifest_path=target.roi_cache_manifest,
                analysis_zarr=target.analysis_zarr,
                crop_run=target.crop_run,
                min_roi_size=int(min_roi_size),
            )
        else:
            cache = supplied_cache
            if cache.manifest_path != target.roi_cache_manifest:
                raise ValueError(
                    f"Supplied cache binding for {target.target_id!r} points to "
                    f"{cache.manifest_path}, expected {target.roi_cache_manifest}."
                )
            if target.crop_run is not None and cache.crop_run != target.crop_run:
                raise ValueError(
                    f"Supplied cache binding for {target.target_id!r} uses crop run "
                    f"{cache.crop_run!r}, expected {target.crop_run!r}."
                )
        input_capability = validate_keypoint_input_dag(
            analysis_zarr=target.analysis_zarr,
            cache=cache,
            min_roi_size=int(min_roi_size),
        )
        validate_registered_geometry_crop_authority(
            analysis_zarr=target.analysis_zarr,
            crop_run=cache.crop_run,
            registered_gate_requirement=gate_requirement,
            registered_gate_run=gate_run,
        )
        native_shape = (int(cache.shape[1]), int(cache.shape[2]))
        model_input_runtime = model_contract.plan_for_native_shape(native_shape)
        if input_mode != "model-contract" and input_mode != model_input_runtime.input_mode:
            raise ValueError(
                "Requested input mode disagrees with the exact model-input contract: "
                f"requested={input_mode!r}, contract={model_input_runtime.input_mode!r}."
            )
        if (
            model_input_stride is not None
            and model_input_stride != model_input_runtime.model_stride
        ):
            raise ValueError(
                "Requested model stride disagrees with the model-input contract: "
                f"requested={model_input_stride}, contract={model_input_runtime.model_stride}."
            )
        model_input_transform = model_input_runtime.transform
        (
            expected_keypoint,
            expected_quality,
            expected_refined,
            expected_body_frame,
        ) = _refuse_output_collisions(
            target=target,
            run_names=run_names,
        )
        prediction_job = build_prediction_job(
            workflow_id=resolved_run_label,
            target_id=target.target_id,
            recording_dir=target.recording_dir,
            analysis_zarr=target.analysis_zarr,
            registry_path=resolved_registry,
            repo=resolved_repo,
            palette_commit=normalized_palette_commit,
            run_root=resolved_run_root,
            run_names=run_names,
            model=model,
            cache=cache,
            model_input_runtime=model_input_runtime,
            pose_schema=pose_schema,
            batch_size=int(batch_size),
            device=device,
            progress_every_batches=int(progress_every_batches),
            resources=prediction_resources,
        )
        if cache.producer_job_key is not None:
            prediction_job = replace(
                prediction_job,
                dependency=LsfDependency((cache.producer_job_key,)),
            )
        refinement_job = build_refinement_job(
            workflow_id=resolved_run_label,
            target_id=target.target_id,
            analysis_zarr=target.analysis_zarr,
            repo=resolved_repo,
            palette_commit=normalized_palette_commit,
            run_root=resolved_run_root,
            run_names=run_names,
            resources=refinement_resources,
            prediction_job=prediction_job,
            crop_run=cache.crop_run,
            recording_identity=target.recording_id,
        )
        jobs.extend((prediction_job, refinement_job))
        finalization_result = (
            resolved_run_root
            / "finalization"
            / f"{safe_component(target.target_id, default='target', max_length=56)}.json"
        )
        planned_targets.append(
            PlannedWholeRecordingTarget(
                target=target,
                model=model,
                cache=cache,
                model_input_contract=model_contract,
                model_input_runtime=model_input_runtime,
                model_input_transform=model_input_transform,
                model_input_stride=model_input_runtime.model_stride,
                input_capability=input_capability,
                run_names=run_names,
                expected_keypoint_group=expected_keypoint,
                expected_keypoint_quality_group=expected_quality,
                expected_refined_keypoint_group=expected_refined,
                expected_body_frame_group=expected_body_frame,
                finalization_result=finalization_result,
                prediction_job_key=prediction_job.job_key,
                refinement_job_key=refinement_job.job_key,
            )
        )

    if not planned_targets or selected_contract is None:
        raise ValueError("Whole-recording keypoint target manifest is empty.")
    effective_model_input_stride = selected_contract.model_stride

    finalizer_job_key = "registry_finalize"
    finalizer_job_name = safe_component(
        f"kp_registry_{resolved_run_label}",
        default="keypoint_registry_finalize",
        max_length=120,
    )
    finalizer_worker = (
        str(resolved_repo / "scripts" / "py"),
        "-m",
        "fisheye.cluster.keypoints.registry_finalize",
        str(resolved_run_root),
        "--registry",
        str(resolved_registry),
        "--output-json",
        str(
            resolved_run_root
            / "registry"
            / f"registry_finalizer.{RUNTIME_JOB_ID_TOKEN}.json"
        ),
    )
    finalizer_command = build_runtime_command(
        finalizer_worker,
        status_path_template=(
            resolved_run_root
            / "status"
            / f"registry_finalizer.{RUNTIME_JOB_ID_TOKEN}.json"
        ),
        workflow_id=resolved_run_label,
        family="keypoints.whole_recording",
        job_key=finalizer_job_key,
        stage="candidate_validation",
        cwd=resolved_repo,
        environment_overrides={
            "PALETTE_REPO": str(resolved_repo),
            "PALETTE_COMMIT": normalized_palette_commit,
        },
        expected_output_templates=(
            str(
                resolved_run_root
                / "registry"
                / f"registry_finalizer.{RUNTIME_JOB_ID_TOKEN}.json"
            ),
        ),
        python_launcher=(str(resolved_repo / "scripts" / "py"),),
    )
    refinement_job_keys = tuple(
        target.refinement_job_key for target in planned_targets
    )
    jobs.append(
        LsfJob(
            job_key=finalizer_job_key,
            job_name=finalizer_job_name,
            command=finalizer_command,
            resources=finalizer_resources,
            stdout_path=resolved_run_root / "logs" / f"{finalizer_job_name}.%J.out",
            stderr_path=resolved_run_root / "logs" / f"{finalizer_job_name}.%J.err",
            dependency=LsfDependency(refinement_job_keys),
            metadata={
                "target_count": len(planned_targets),
                "candidate_validation_only": True,
                "selector_activation": False,
                "registry_mutation": False,
                "registry": str(resolved_registry),
                "palette_repo": str(resolved_repo),
                "palette_commit": normalized_palette_commit,
            },
        )
    )
    workflow = LsfWorkflow(
        workflow_id=resolved_run_label,
        family="keypoints.whole_recording",
        jobs=tuple(jobs),
        metadata={
            "plan_schema": PLAN_SCHEMA,
            "manifest_path": str(resolved_manifest),
            "manifest_sha256": manifest_sha256,
            "palette_repo": str(resolved_repo),
            "palette_commit": normalized_palette_commit,
            "model_input_contract": selected_contract.to_json(),
            "model_input_stride": effective_model_input_stride,
            "target_count": len(planned_targets),
            "keypoint_storage": keypoint_storage,
            "finalization_execution": finalization_execution,
            "registered_dish_geometry": {
                "gate_requirement": gate_requirement,
                "gate_run": gate_run,
            },
        },
    )
    return WholeRecordingWorkflowPlan(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_sha256,
        run_label=resolved_run_label,
        repo=resolved_repo,
        palette_commit=normalized_palette_commit,
        registry=resolved_registry,
        run_root=resolved_run_root,
        model_set_id=str(model_set_id),
        model_run_id=str(model_run_id),
        model_input_contract=selected_contract,
        pose_schema=str(pose_schema),
        min_roi_size=int(min_roi_size),
        model_input_stride=effective_model_input_stride,
        keypoint_storage=keypoint_storage,
        finalization_execution=finalization_execution,
        registered_gate_requirement=gate_requirement,
        registered_gate_run=gate_run,
        targets=tuple(planned_targets),
        registry_finalizer_job_key=finalizer_job_key,
        lsf_workflow=workflow,
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def materialize_plan_bundle(plan: WholeRecordingWorkflowPlan) -> dict[str, Any]:
    """Persist the reviewed target list and exact DAG before any submission."""

    plan.run_root.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "progress", "status", "registry"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    plan_path = plan.run_root / "plan.json"
    payload = plan.to_json()
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"Run root already contains a different plan: {plan_path}"
            )
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(plan.run_root / "lsf_plan.json", plan.lsf_workflow.to_json())
    write_json_snapshot(
        plan.run_root / "targets.normalized.json",
        {
            "schema": NORMALIZED_TARGETS_SCHEMA,
            "source_manifest": str(plan.manifest_path),
            "source_manifest_sha256": plan.manifest_sha256,
            "target_count": len(plan.targets),
            "targets": [target.to_json() for target in plan.targets],
        },
    )
    _write_text_atomic(
        plan.run_root / "zarr_paths.txt",
        "".join(f"{target.target.analysis_zarr}\n" for target in plan.targets),
    )
    return payload


def apply_plan(
    plan: WholeRecordingWorkflowPlan,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Persist and submit the exact workflow.  This is the only bsub path."""

    if (plan.run_root / "lsf_submission.json").exists():
        raise FileExistsError(
            f"Submission evidence already exists: {plan.run_root / 'lsf_submission.json'}"
        )
    materialize_plan_bundle(plan)
    return submit_lsf_workflow(
        plan.lsf_workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=plan.run_root / "lsf_submission.json",
        runner=runner,
    )


def _print_plan(plan: WholeRecordingWorkflowPlan) -> None:
    print("Whole-recording keypoint/refinement workflow dry-run")
    print(f"  manifest: {plan.manifest_path}")
    print(f"  run_label: {plan.run_label}")
    print(f"  run_root: {plan.run_root}")
    print(f"  targets: {len(plan.targets)}")
    print(f"  model: {plan.model_set_id} / {plan.model_run_id}")
    print(f"  model_input_contract: {plan.model_input_contract.path}")
    print(
        "  model_input: "
        f"source={plan.model_input_contract.training_source_shape_hw}, "
        f"network={plan.model_input_contract.network_shape_hw}, "
        f"mode={plan.model_input_contract.input_mode}"
    )
    print(f"  pose_schema: {plan.pose_schema}")
    print(f"  minimum ROI size: {plan.min_roi_size}x{plan.min_roi_size}")
    print(f"  keypoint_storage: {plan.keypoint_storage['effective']}")
    print()
    print("DAG")
    for target in plan.targets:
        print(
            f"  {target.prediction_job_key} -> {target.refinement_job_key} "
            f"[{target.target.analysis_zarr}]"
        )
    print(
        "  "
        + ", ".join(target.refinement_job_key for target in plan.targets)
        + f" -> {plan.registry_finalizer_job_key}"
    )
    print()
    print("bsub command templates (no jobs submitted)")
    for job in plan.lsf_workflow.topological_jobs():
        print(f"  {shell_join(build_bsub_command(job))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--palette-repo",
        "--repo",
        dest="repo",
        type=Path,
        default=DEFAULT_GROUPS_REPO,
        help="Absolute commit-pinned Palette deployment (legacy alias: --repo).",
    )
    parser.add_argument(
        "--palette-commit",
        required=True,
        help="Full commit checked out at --palette-repo.",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-set-id", required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument(
        "--registered-gate-requirement",
        choices=("off", "if_available", "required"),
        default="off",
    )
    parser.add_argument("--registered-gate-run")
    parser.add_argument(
        "--model-input-contract",
        type=Path,
        required=True,
        help=(
            "Digest-bound model package input contract. It owns submitted "
            "extent, network imgsz, runtime adapter, and stride."
        ),
    )
    parser.add_argument("--pose-schema", default="traditional_v2")
    parser.add_argument(
        "--min-roi-size",
        type=int,
        default=DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
        help="Minimum cache/crop height and width for zebrafish keypoints.",
    )
    parser.add_argument("--batch-size-kp", type=int, default=256)
    keypoint_storage_group = parser.add_mutually_exclusive_group()
    keypoint_storage_group.add_argument(
        "--keypoint-roi-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
        help=(
            "Legacy compatibility input only; strict v2 publication ignores "
            "row-count shard overrides and plans from uncompressed bytes."
        ),
    )
    keypoint_storage_group.add_argument(
        "--no-keypoint-sharding",
        action="store_const",
        dest="keypoint_roi_shard_rows",
        const=None,
        help=(
            "Legacy compatibility input only; strict v2 publication remains "
            "byte-planned and access-aware."
        ),
    )
    parser.add_argument(
        "--keypoint-frame-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
        help=(
            "Legacy compatibility input only; strict v2 frame indexes use the "
            "shared byte planner."
        ),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--input-mode",
        choices=("model-contract", "numpy-list", "tensor", "auto"),
        default="model-contract",
        help="Optional assertion; model-contract selects the bound runtime mode.",
    )
    parser.add_argument(
        "--model-input-stride",
        type=int,
        default=None,
        help=(
            "Optional assertion against the contract-owned maximum model stride."
        ),
    )
    parser.add_argument("--progress-every-batches", type=int, default=1)

    parser.add_argument("--prediction-queue", default="gpu_l4")
    parser.add_argument("--prediction-ncores", type=int, default=4)
    parser.add_argument("--prediction-mem-gb", type=int, default=32)
    parser.add_argument("--prediction-gpus", type=int, default=1)
    parser.add_argument("--prediction-walltime", default=None)

    parser.add_argument("--refine-queue", default="short")
    parser.add_argument("--refine-ncores", type=int, default=4)
    parser.add_argument("--refine-mem-gb", type=int, default=16)
    parser.add_argument("--refine-walltime", default="1:00")
    parser.add_argument(
        "--refine-chunk-size",
        type=int,
        default=2048,
        help="Legacy compatibility input; ignored by the strict v2 finalizer.",
    )
    parser.add_argument(
        "--refine-scheduler",
        choices=("processes", "threads", "distributed"),
        default="threads",
        help="Legacy compatibility input; ignored by the strict v2 finalizer.",
    )
    parser.add_argument(
        "--refine-num-workers",
        type=int,
        default=4,
        help="Legacy compatibility input; ignored by the strict v2 finalizer.",
    )
    parser.add_argument(
        "--refine-memory-limit",
        default=None,
        help="Legacy compatibility input; ignored by the strict v2 finalizer.",
    )

    parser.add_argument("--finalizer-queue", default="short")
    parser.add_argument("--finalizer-ncores", type=int, default=1)
    parser.add_argument("--finalizer-mem-gb", type=int, default=8)
    parser.add_argument("--finalizer-walltime", default="1:00")
    parser.add_argument("--json", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(
        manifest_path=args.manifest,
        run_label=args.run_label,
        repo=args.repo,
        palette_commit=args.palette_commit,
        registry=args.registry,
        run_root=args.run_root,
        model_set_id=args.model_set_id,
        model_run_id=args.model_run_id,
        model_input_contract_path=args.model_input_contract,
        pose_schema=args.pose_schema,
        min_roi_size=int(args.min_roi_size),
        batch_size=int(args.batch_size_kp),
        device=args.device,
        input_mode=args.input_mode,
        model_input_stride=(
            int(args.model_input_stride)
            if args.model_input_stride is not None
            else None
        ),
        progress_every_batches=int(args.progress_every_batches),
        keypoint_roi_shard_rows=args.keypoint_roi_shard_rows,
        keypoint_frame_shard_rows=int(args.keypoint_frame_shard_rows),
        prediction_resources=LsfResources(
            queue=args.prediction_queue,
            ncores=int(args.prediction_ncores),
            mem_gb=int(args.prediction_mem_gb),
            gpus=int(args.prediction_gpus),
            walltime=args.prediction_walltime,
        ),
        refinement_resources=LsfResources(
            queue=args.refine_queue,
            ncores=int(args.refine_ncores),
            mem_gb=int(args.refine_mem_gb),
            walltime=args.refine_walltime,
        ),
        refine_chunk_size=int(args.refine_chunk_size),
        refine_scheduler=args.refine_scheduler,
        refine_num_workers=int(args.refine_num_workers),
        refine_memory_limit=args.refine_memory_limit,
        finalizer_resources=LsfResources(
            queue=args.finalizer_queue,
            ncores=int(args.finalizer_ncores),
            mem_gb=int(args.finalizer_mem_gb),
            walltime=args.finalizer_walltime,
        ),
        registered_gate_requirement=args.registered_gate_requirement,
        registered_gate_run=args.registered_gate_run,
    )
    if args.apply:
        result = apply_plan(plan)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("Submitted whole-recording keypoint/refinement workflow")
            print(f"  run_root: {plan.run_root}")
            print(f"  jobs: {len(plan.lsf_workflow.jobs)}")
            print(f"  submission: {plan.run_root / 'lsf_submission.json'}")
        return 0
    materialize_plan_bundle(plan)
    if args.json:
        print(json.dumps(plan.to_json(), indent=2, sort_keys=True))
    else:
        _print_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZED_TARGETS_SCHEMA",
    "PLAN_SCHEMA",
    "TARGET_MANIFEST_SCHEMA",
    "PlannedWholeRecordingTarget",
    "WholeRecordingTarget",
    "WholeRecordingWorkflowPlan",
    "apply_plan",
    "build_plan",
    "load_target_manifest",
    "main",
    "materialize_plan_bundle",
]
