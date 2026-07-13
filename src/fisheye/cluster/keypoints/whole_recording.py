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
    resolve_keypoint_storage,
    safe_component,
    validate_flat_roi_cache_binding,
    validate_keypoint_input_dag,
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


TARGET_MANIFEST_SCHEMA = "palette.whole_recording_keypoint_targets.v1"
PLAN_SCHEMA = "palette.whole_recording_keypoint_bsub_plan.v1"
NORMALIZED_TARGETS_SCHEMA = "palette.whole_recording_keypoint_targets.normalized.v1"
DEFAULT_GROUPS_REPO = Path("/groups/johnson/johnsonlab/jeremy/gitrepos/palette")
DEFAULT_REGISTRY = Path(
    "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
)


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
    input_capability: KeypointInputCapability
    run_names: KeypointRunNames
    expected_keypoint_group: Path
    expected_refined_keypoint_group: Path
    prediction_job_key: str
    refinement_job_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target.to_json(),
            "model": self.model.to_json(),
            "cache": self.cache.to_json(),
            "input_capability": self.input_capability.to_json(),
            "run_names": self.run_names.to_json(),
            "expected_keypoint_group": str(self.expected_keypoint_group),
            "expected_refined_keypoint_group": str(
                self.expected_refined_keypoint_group
            ),
            "prediction_job_key": self.prediction_job_key,
            "refinement_job_key": self.refinement_job_key,
        }


@dataclass(frozen=True)
class WholeRecordingWorkflowPlan:
    manifest_path: Path
    manifest_sha256: str
    run_label: str
    repo: Path
    registry: Path
    run_root: Path
    model_set_id: str
    model_run_id: str
    pose_schema: str
    min_roi_size: int
    keypoint_storage: dict[str, Any]
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
            "registry": str(self.registry),
            "run_root": str(self.run_root),
            "model_set_id": self.model_set_id,
            "model_run_id": self.model_run_id,
            "pose_schema": self.pose_schema,
            "min_roi_size": self.min_roi_size,
            "keypoint_storage": self.keypoint_storage,
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
) -> tuple[Path, Path]:
    keypoint_group = target.analysis_zarr / "keypoints_runs" / run_names.keypoint_run
    refined_group = (
        target.analysis_zarr
        / "refined_keypoints_runs"
        / run_names.refined_keypoint_run
    )
    collisions = [path for path in (keypoint_group, refined_group) if path.exists()]
    if collisions:
        raise FileExistsError(
            f"Planned output already exists for {target.target_id!r}: "
            + ", ".join(str(path) for path in collisions)
        )
    return keypoint_group, refined_group


def build_plan(
    *,
    manifest_path: Path,
    run_label: str,
    repo: Path,
    registry: Path,
    run_root: Path,
    model_set_id: str,
    model_run_id: str,
    pose_schema: str,
    min_roi_size: int,
    batch_size: int,
    device: str,
    input_mode: str,
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
) -> WholeRecordingWorkflowPlan:
    resolved_manifest = manifest_path.expanduser().resolve()
    resolved_repo = repo.expanduser().resolve()
    resolved_registry = registry.expanduser().resolve()
    resolved_run_root = run_root.expanduser().resolve()
    resolved_run_label = safe_component(run_label, default="keypoints")
    if not resolved_repo.is_dir() or not (resolved_repo / "scripts" / "py").is_file():
        raise FileNotFoundError(
            f"Palette repository or scripts/py not found: {resolved_repo}"
        )
    if not resolved_registry.is_file():
        raise FileNotFoundError(f"Palette registry not found: {resolved_registry}")
    if not str(model_set_id).strip() or not str(model_run_id).strip():
        raise ValueError("Both model_set_id and model_run_id are required.")
    if int(batch_size) <= 0 or int(progress_every_batches) <= 0:
        raise ValueError("Batch size and progress interval must be positive.")
    if int(min_roi_size) <= 0:
        raise ValueError("Minimum zebrafish ROI size must be positive.")
    if int(refine_chunk_size) <= 0 or int(refine_num_workers) <= 0:
        raise ValueError("Refinement chunk size and worker count must be positive.")
    keypoint_storage = resolve_keypoint_storage(
        roi_shard_rows=keypoint_roi_shard_rows,
        frame_shard_rows=keypoint_frame_shard_rows,
    )

    manifest_bytes = resolved_manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    targets = load_target_manifest(resolved_manifest)
    run_names = build_keypoint_run_names(resolved_run_label)
    jobs: list[LsfJob] = list(upstream_jobs)
    planned_targets: list[PlannedWholeRecordingTarget] = []
    model_identity: tuple[str, str, str, str] | None = None

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
        expected_keypoint, expected_refined = _refuse_output_collisions(
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
            run_root=resolved_run_root,
            run_names=run_names,
            model=model,
            cache=cache,
            pose_schema=pose_schema,
            batch_size=int(batch_size),
            device=device,
            input_mode=input_mode,
            progress_every_batches=int(progress_every_batches),
            keypoint_roi_shard_rows=keypoint_roi_shard_rows,
            keypoint_frame_shard_rows=int(keypoint_frame_shard_rows),
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
            run_root=resolved_run_root,
            run_names=run_names,
            chunk_size=int(refine_chunk_size),
            scheduler=refine_scheduler,
            num_workers=int(refine_num_workers),
            memory_limit=refine_memory_limit,
            resources=refinement_resources,
            prediction_job=prediction_job,
        )
        jobs.extend((prediction_job, refinement_job))
        planned_targets.append(
            PlannedWholeRecordingTarget(
                target=target,
                model=model,
                cache=cache,
                input_capability=input_capability,
                run_names=run_names,
                expected_keypoint_group=expected_keypoint,
                expected_refined_keypoint_group=expected_refined,
                prediction_job_key=prediction_job.job_key,
                refinement_job_key=refinement_job.job_key,
            )
        )

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
        "--apply",
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
        stage="registry_finalization",
        cwd=resolved_repo,
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
                "keypoint_run": run_names.keypoint_run,
                "refined_keypoint_run": run_names.refined_keypoint_run,
                "registry": str(resolved_registry),
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
            "target_count": len(planned_targets),
            "keypoint_storage": keypoint_storage,
        },
    )
    return WholeRecordingWorkflowPlan(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_sha256,
        run_label=resolved_run_label,
        repo=resolved_repo,
        registry=resolved_registry,
        run_root=resolved_run_root,
        model_set_id=str(model_set_id),
        model_run_id=str(model_run_id),
        pose_schema=str(pose_schema),
        min_roi_size=int(min_roi_size),
        keypoint_storage=keypoint_storage,
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
    parser.add_argument("--repo", type=Path, default=DEFAULT_GROUPS_REPO)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-set-id", required=True)
    parser.add_argument("--model-run-id", required=True)
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
            "Outer rows for indexed-sharded ROI arrays "
            f"(default: {DEFAULT_KEYPOINT_ROI_SHARD_ROWS})."
        ),
    )
    keypoint_storage_group.add_argument(
        "--no-keypoint-sharding",
        action="store_const",
        dest="keypoint_roi_shard_rows",
        const=None,
        help="Use ordinary chunks for keypoint outputs.",
    )
    parser.add_argument(
        "--keypoint-frame-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
        help=(
            "Outer rows for indexed-sharded frame arrays "
            f"(default: {DEFAULT_KEYPOINT_FRAME_SHARD_ROWS})."
        ),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--input-mode",
        choices=("numpy-list", "tensor", "auto"),
        default="tensor",
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
    parser.add_argument("--refine-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--refine-scheduler",
        choices=("processes", "threads", "distributed"),
        default="threads",
    )
    parser.add_argument("--refine-num-workers", type=int, default=4)
    parser.add_argument("--refine-memory-limit", default=None)

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
        registry=args.registry,
        run_root=args.run_root,
        model_set_id=args.model_set_id,
        model_run_id=args.model_run_id,
        pose_schema=args.pose_schema,
        min_roi_size=int(args.min_roi_size),
        batch_size=int(args.batch_size_kp),
        device=args.device,
        input_mode=args.input_mode,
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
