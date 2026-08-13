"""Plan registry-discovered whole-video detection as one bounded LSF array."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.clipped_detection import (
    DetectionModelSpec,
    RawDetectionFragmentInputs,
    RawDetectionWorkUnitSpec,
    build_whole_video_raw_detection_cohort_fragment,
)
from fisheye.cluster.clipped_inference import build_ssh_bsub_runner
from fisheye.cluster.crop_snapshot import (
    CropSnapshotFragmentInputs,
    CropSnapshotFragmentOutputs,
    build_crop_snapshot_fragment,
)
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfWorkflow,
    compose_lsf_workflow,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.recording_layout import (
    RecordingTarget,
    whole_video_recording_target,
)
from fisheye.cluster.recording_detection_postprocess import (
    REGISTERED_GATE_REQUIREMENTS,
    RecordingDetectionPostprocessOutputs,
    RecordingDetectionPostprocessInputs,
    build_recording_detection_postprocess_fragment,
)
from fisheye.shared.crop_defaults import DEFAULT_ZEBRAFISH_CROP_SIZE_PX
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.model_resolution import (
    load_candidates,
    load_target_profile,
    resolve_recording_id,
    verify_deployment_artifact_content,
)


PLAN_SCHEMA = "palette.whole_video_detection_cohort_plan.v1"
FAMILY = "whole_video_detection"
DEFAULT_REPO = Path("/groups/johnson/johnsonlab/jeremy/gitrepos/palette")
AUTHORITATIVE_FULL_FRAME_ROLE = "ingest_authoritative_full_frame"


@dataclass(frozen=True)
class RegistryWholeVideoTarget:
    """One exact registry dataset and its authoritative full-frame stream."""

    dataset_id: str
    target: RecordingTarget
    stream_key: str
    stream_role: str

    def to_json(self) -> dict[str, object]:
        unit = self.target.work_units[0]
        return {
            "dataset_id": self.dataset_id,
            "target_id": self.target.target_id,
            "recording_id": self.target.recording_id,
            "recording_dir": str(self.target.recording_dir),
            "analysis_zarr": str(self.target.analysis_zarr),
            "camera_serial": unit.camera_serial,
            "video_path": str(unit.video_path),
            "stream_key": self.stream_key,
            "stream_role": self.stream_role,
            "frame_mapping": unit.frame_mapping.to_json(),
        }


@dataclass(frozen=True)
class WholeVideoDetectionCohortPlan:
    """Immutable planner result suitable for dry-run evidence or submission."""

    run_label: str
    workflow_id: str
    repo: Path
    registry_path: Path
    run_root: Path
    targets: tuple[RegistryWholeVideoTarget, ...]
    model: DetectionModelSpec
    detect_run: str
    quality_run: str | None
    refined_run: str | None
    crop_run: str | None
    registered_gate_requirement: str
    registered_gate_run: str | None
    selection_policy_id: str
    postprocess_outputs: tuple[RecordingDetectionPostprocessOutputs, ...]
    crop_outputs: tuple[CropSnapshotFragmentOutputs, ...]
    max_concurrent: int
    lsf_workflow: LsfWorkflow

    def to_json(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "run_label": self.run_label,
            "workflow_id": self.workflow_id,
            "repo": str(self.repo),
            "registry": str(self.registry_path),
            "run_root": str(self.run_root),
            "target_count": len(self.targets),
            "targets": [target.to_json() for target in self.targets],
            "model": {
                "set_id": self.model.set_id,
                "run_id": self.model.run_id,
                "path": str(self.model.path),
                "sha256": self.model.sha256,
            },
            "detect_run": self.detect_run,
            "quality_run": self.quality_run,
            "refined_run": self.refined_run,
            "crop_run": self.crop_run,
            "registered_dish_geometry": {
                "gate_requirement": self.registered_gate_requirement,
                "gate_run": self.registered_gate_run,
                "selection_policy_id": self.selection_policy_id,
            },
            "postprocess_outputs": [
                output.to_json() for output in self.postprocess_outputs
            ],
            "crop_outputs": [output.to_json() for output in self.crop_outputs],
            "scheduler": {
                "execution_mode": "lsf_array",
                "max_concurrent": self.max_concurrent,
            },
            "lsf_workflow": self.lsf_workflow.to_json(),
        }


def _recording_dir_for_zarr(zarr_path: Path) -> Path:
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    raise ValueError(
        "Whole-video production requires the canonical "
        f"<recording>/zarr/<archive>.zarr layout: {zarr_path}"
    )


def discover_registry_whole_video_targets(
    registry: Registry,
    *,
    recording_ids: Sequence[str] = (),
    path_contains: str | None = None,
    zarr_use: str = "analysis",
    limit: int | None = None,
) -> tuple[RegistryWholeVideoTarget, ...]:
    """Discover exact analysis Zarr/full-video pairs without opening Zarr."""

    requested = tuple(
        str(value).strip() for value in recording_ids if str(value).strip()
    )
    if len(set(requested)) != len(requested):
        raise ValueError("Requested recording ids must be unique.")
    if not requested and not str(path_contains or "").strip():
        raise ValueError(
            "Registry discovery requires explicit recording ids or --path-contains."
        )
    rows = registry.query_datasets(
        zarr_use=zarr_use,
        path_contains=path_contains,
        status="active",
        require_recording=True,
        limit=limit,
    )
    if requested:
        requested_set = set(requested)
        rows = [row for row in rows if str(row["recording_id"]) in requested_set]
        missing = requested_set - {str(row["recording_id"]) for row in rows}
        if missing:
            raise ValueError(
                "Registry discovery did not resolve requested recording ids: "
                + ", ".join(sorted(missing))
            )
    if not rows:
        raise ValueError(
            "Registry discovery selected no whole-video analysis datasets."
        )

    rows_by_recording: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_recording.setdefault(str(row["recording_id"]), []).append(row)
    duplicates = {
        recording_id: values
        for recording_id, values in rows_by_recording.items()
        if len(values) != 1
    }
    if duplicates:
        raise ValueError(
            "Whole-video planning requires one active analysis dataset per recording; "
            f"ambiguous recordings: {sorted(duplicates)!r}."
        )

    discovered: list[RegistryWholeVideoTarget] = []
    for index, recording_id in enumerate(sorted(rows_by_recording)):
        row = rows_by_recording[recording_id][0]
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"])).expanduser().resolve()
        if not (zarr_path / "zarr.json").is_file():
            raise FileNotFoundError(
                f"Registered analysis Zarr is not a live Zarr v3 root: {zarr_path}"
            )
        streams = registry.query_acquisition_video_streams_current(
            dataset_id=dataset_id,
            recording_id=recording_id,
            output_kind="full",
            availability_status="ok",
            require_video=True,
        )
        streams = [
            stream
            for stream in streams
            if str(stream["role"] or "") == AUTHORITATIVE_FULL_FRAME_ROLE
        ]
        if len(streams) != 1:
            raise ValueError(
                f"Recording {recording_id!r} requires exactly one available "
                f"{AUTHORITATIVE_FULL_FRAME_ROLE!r} stream; found {len(streams)}."
            )
        stream = streams[0]
        recording_dir = _recording_dir_for_zarr(zarr_path)
        video_value = str(stream["video_path"] or "").strip()
        if not video_value:
            raise ValueError(
                f"Registry full-video path is missing for {recording_id!r}."
            )
        declared_video_path = Path(video_value).expanduser()
        if declared_video_path.is_absolute():
            video_path = declared_video_path.resolve()
        else:
            video_path = (recording_dir / declared_video_path).resolve()
            try:
                video_path.relative_to(recording_dir)
            except ValueError as exc:
                raise ValueError(
                    "Registry relative full-video path escapes the recording root for "
                    f"{recording_id!r}: {declared_video_path}"
                ) from exc
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Registry full-video path is not live for {recording_id!r}: {video_path}"
            )
        camera_serial = str(
            stream["camera_id"] or row["camera_serial"] or row["camera_id"] or ""
        ).strip()
        if not camera_serial:
            raise ValueError(f"Recording {recording_id!r} has no camera serial.")
        target_id = safe_component(
            recording_id,
            default=f"recording_{index:04d}",
            max_length=80,
        )
        target = whole_video_recording_target(
            target_id=target_id,
            recording_id=recording_id,
            recording_dir=recording_dir,
            analysis_zarr=zarr_path,
            video_path=video_path,
            camera_serial=camera_serial,
        )
        discovered.append(
            RegistryWholeVideoTarget(
                dataset_id=dataset_id,
                target=target,
                stream_key=str(stream["stream_key"]),
                stream_role=str(stream["role"]),
            )
        )

    target_ids = [item.target.target_id for item in discovered]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(
            "Registry recording ids collide after safe target normalization."
        )
    return tuple(discovered)


def resolve_detection_model_for_targets(
    registry: Registry,
    targets: Sequence[RegistryWholeVideoTarget],
    *,
    set_id: str,
    run_id: str,
) -> DetectionModelSpec:
    """Require one exact successful, content-pinned model for the cohort."""

    identities: set[tuple[str, str, str, str]] = set()
    for item in targets:
        target = item.target
        resolved_recording_id = resolve_recording_id(
            registry,
            recording_id=target.recording_id,
            recording_dir=target.recording_dir,
        )
        profile = load_target_profile(registry, resolved_recording_id)
        candidates = load_candidates(
            registry,
            target=profile,
            task="detect",
            set_id_filter=set_id,
            include_non_success=False,
        )
        exact = [candidate for candidate in candidates if candidate.run_id == run_id]
        if len(exact) != 1:
            raise ValueError(
                f"Expected one successful detect model {set_id!r}/{run_id!r} for "
                f"{target.recording_id!r}; found {len(exact)}."
            )
        candidate = exact[0]
        if not candidate.model_sha256:
            raise ValueError(f"Registered detection model {run_id!r} has no SHA-256.")
        verification = verify_deployment_artifact_content(
            {
                "model_path": candidate.model_path,
                "model_sha256": candidate.model_sha256,
            },
            artifact="model",
            role="detect_deployment_model",
        )
        if verification.status != "match":
            raise ValueError(
                "Registered detection model is not content-pinned: "
                + json.dumps(verification.to_dict(), sort_keys=True)
            )
        identities.add(
            (
                str(candidate.set_id),
                str(candidate.run_id),
                str(Path(candidate.model_path).expanduser().resolve()),
                str(candidate.model_sha256),
            )
        )
    if len(identities) != 1:
        raise ValueError(
            "The selected recordings do not resolve one common detection model: "
            f"{sorted(identities)!r}."
        )
    resolved_set, resolved_run, path, sha256 = next(iter(identities))
    return DetectionModelSpec(
        set_id=resolved_set,
        run_id=resolved_run,
        path=Path(path),
        sha256=sha256,
    )


def build_plan(
    *,
    registry_path: Path,
    repo: Path,
    run_root: Path,
    run_label: str,
    detection_set_id: str,
    detection_run_id: str,
    recording_ids: Sequence[str] = (),
    path_contains: str | None = None,
    zarr_use: str = "analysis",
    limit: int | None = None,
    max_concurrent: int = 8,
    include_postprocess: bool = False,
    include_crop: bool = False,
    registered_gate_requirement: str = "off",
    registered_gate_run: str | None = None,
    selection_policy_id: str = "manual_review_only_v1",
) -> WholeVideoDetectionCohortPlan:
    """Resolve registry targets/model and build one immutable array plan."""

    resolved_registry = registry_path.expanduser().resolve()
    resolved_repo = repo.expanduser().resolve()
    resolved_run_root = run_root.expanduser().resolve()
    if int(max_concurrent) <= 0:
        raise ValueError("max_concurrent must be positive.")
    if type(include_postprocess) is not bool:
        raise TypeError("include_postprocess must be an exact bool.")
    if type(include_crop) is not bool:
        raise TypeError("include_crop must be an exact bool.")
    gate_requirement = str(registered_gate_requirement).strip()
    if gate_requirement not in REGISTERED_GATE_REQUIREMENTS:
        raise ValueError(
            "registered_gate_requirement must be off, if_available, or required."
        )
    gate_run = str(registered_gate_run or "").strip() or None
    if gate_requirement == "required" and gate_run is None:
        raise ValueError("Required registered geometry needs an exact gate run.")
    if gate_requirement != "off":
        include_postprocess = True
        include_crop = True
    if include_crop:
        include_postprocess = True
    policy_id = str(selection_policy_id).strip()
    if policy_id not in {"manual_review_only_v1", "corroborated_acquisition_v1"}:
        raise ValueError("Unsupported registered geometry selection policy id.")
    if not resolved_registry.is_file():
        raise FileNotFoundError(f"Registry does not exist: {resolved_registry}")
    if not (resolved_repo / "scripts" / "py").is_file():
        raise FileNotFoundError(f"Palette checkout lacks scripts/py: {resolved_repo}")
    safe_label = safe_component(run_label, default="whole_video_detect", max_length=64)
    workflow_id = safe_component(
        f"whole_video_detect_{safe_label}",
        default="whole_video_detect",
        max_length=96,
    )
    detect_run = safe_component(
        f"detect_{safe_label}",
        default="detect_whole_video",
        max_length=120,
    )
    quality_run = (
        safe_component(
            f"detect_quality_{safe_label}",
            default="detect_quality_whole_video",
            max_length=120,
        )
        if include_postprocess
        else None
    )
    refined_run = (
        safe_component(
            f"refined_detect_{safe_label}",
            default="refined_detect_whole_video",
            max_length=120,
        )
        if include_postprocess
        else None
    )
    crop_run = (
        safe_component(
            f"crop_{safe_label}",
            default="crop_whole_video",
            max_length=120,
        )
        if include_crop
        else None
    )

    registry = Registry(resolved_registry)
    try:
        targets = discover_registry_whole_video_targets(
            registry,
            recording_ids=recording_ids,
            path_contains=path_contains,
            zarr_use=zarr_use,
            limit=limit,
        )
        model = resolve_detection_model_for_targets(
            registry,
            targets,
            set_id=detection_set_id,
            run_id=detection_run_id,
        )
    finally:
        registry.close()

    fragment_inputs: list[RawDetectionFragmentInputs] = []
    for item in targets:
        target = item.target
        detect_group = f"detect_runs/{detect_run}"
        if (target.analysis_zarr / detect_group).exists():
            raise FileExistsError(
                "Whole-video cohort planning refuses existing detection output: "
                f"{target.analysis_zarr / detect_group}"
            )
        fragment_inputs.append(
            RawDetectionFragmentInputs(
                workflow_id=workflow_id,
                family=FAMILY,
                target_label=target.target_id,
                target=target,
                repo=resolved_repo,
                run_root=resolved_run_root,
                work_units=(
                    RawDetectionWorkUnitSpec(
                        work_unit=target.work_units[0],
                        detect_run=detect_run,
                        detect_group_path=detect_group,
                    ),
                ),
                model=model,
                registry_path=resolved_registry,
            )
        )
    cohort = build_whole_video_raw_detection_cohort_fragment(
        fragment_inputs,
        max_concurrent=int(max_concurrent),
    )
    postprocess_modules = ()
    if include_postprocess:
        assert quality_run is not None and refined_run is not None
        outputs_by_target = {output.target_id: output for output in cohort.outputs}
        postprocess_modules = tuple(
            build_recording_detection_postprocess_fragment(
                RecordingDetectionPostprocessInputs(
                    workflow_id=workflow_id,
                    family=FAMILY,
                    target=item.target,
                    repo=resolved_repo,
                    run_root=resolved_run_root,
                    source_detect_run=detect_run,
                    canonicalize_legacy_source=True,
                    canonical_source_run=safe_component(
                        f"{detect_run}_canonical_v3",
                        default="detect_whole_video_canonical_v3",
                        max_length=120,
                    ),
                    quality_run=quality_run,
                    refined_run=refined_run,
                    registered_gate_requirement=gate_requirement,
                    registered_gate_run=gate_run,
                    selection_policy_id=policy_id,
                    upstream_job_keys=(
                        outputs_by_target[item.target.target_id].terminal_job_key,
                    ),
                    required_artifacts=(
                        outputs_by_target[item.target.target_id].artifact_key,
                    ),
                )
            )
            for item in targets
        )
    crop_modules = ()
    if include_crop:
        assert crop_run is not None and refined_run is not None
        postprocess_by_target = {
            module.outputs.target_id: module.outputs
            for module in postprocess_modules
        }
        crop_modules = tuple(
            build_crop_snapshot_fragment(
                CropSnapshotFragmentInputs(
                    workflow_id=workflow_id,
                    family=FAMILY,
                    target_id=item.target.target_id,
                    analysis_zarr=item.target.analysis_zarr,
                    repo=resolved_repo,
                    run_root=resolved_run_root,
                    run_id=crop_run,
                    purpose="zebrafish_keypoints_and_subject_masks",
                    roi_width=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
                    roi_height=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
                    camera_id=item.target.work_units[0].camera_serial,
                    source_refined_run=refined_run,
                    registered_gate_requirement=gate_requirement,
                    registered_gate_run=gate_run,
                    upstream_job_keys=(
                        postprocess_by_target[item.target.target_id].terminal_job_key,
                    ),
                    required_artifacts=(
                        postprocess_by_target[item.target.target_id].artifact_key,
                    ),
                )
            )
            for item in targets
        )
    workflow = compose_lsf_workflow(
        workflow_id=workflow_id,
        family=FAMILY,
        fragments=(
            cohort.fragment,
            *(module.fragment for module in postprocess_modules),
            *(module.fragment for module in crop_modules),
        ),
        metadata={
            "workflow_scope": "registry_discovered_whole_video_raw_detection",
            "target_count": len(targets),
            "model": {
                "set_id": model.set_id,
                "run_id": model.run_id,
                "path": str(model.path),
                "sha256": model.sha256,
            },
            "outputs": [output.to_json() for output in cohort.outputs],
            "postprocess_outputs": [
                module.outputs.to_json() for module in postprocess_modules
            ],
            "crop_outputs": [module.outputs.to_json() for module in crop_modules],
            "registered_dish_geometry": {
                "gate_requirement": gate_requirement,
                "gate_run": gate_run,
                "selection_policy_id": policy_id,
            },
        },
    )
    return WholeVideoDetectionCohortPlan(
        run_label=safe_label,
        workflow_id=workflow_id,
        repo=resolved_repo,
        registry_path=resolved_registry,
        run_root=resolved_run_root,
        targets=targets,
        model=model,
        detect_run=detect_run,
        quality_run=quality_run,
        refined_run=refined_run,
        crop_run=crop_run,
        registered_gate_requirement=gate_requirement,
        registered_gate_run=gate_run,
        selection_policy_id=policy_id,
        postprocess_outputs=tuple(
            module.outputs for module in postprocess_modules
        ),
        crop_outputs=tuple(module.outputs for module in crop_modules),
        max_concurrent=min(int(max_concurrent), len(targets)),
        lsf_workflow=workflow,
    )


def materialize_plan_bundle(plan: WholeVideoDetectionCohortPlan) -> dict[str, object]:
    """Persist immutable dry-run evidence consumed by the LSF task-group runner."""

    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"Run root contains a different immutable plan: {plan_path}"
            )
        if (
            not lsf_path.is_file()
            or json.loads(lsf_path.read_text(encoding="utf-8"))
            != plan.lsf_workflow.to_json()
        ):
            raise FileExistsError(
                f"Run root has mismatched LSF plan evidence: {lsf_path}"
            )
        return existing
    for name in ("logs", "status", "targets"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.lsf_workflow.to_json())
    return payload


def apply_plan(
    plan: WholeVideoDetectionCohortPlan,
    *,
    submit_host: str,
) -> dict[str, Any]:
    """Materialize then submit through the Citrus login poller."""

    submission_path = plan.run_root / "lsf_submission.json"
    if submission_path.exists():
        raise FileExistsError(f"Submission evidence already exists: {submission_path}")
    materialize_plan_bundle(plan)
    return submit_lsf_workflow(
        plan.lsf_workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=submission_path,
        runner=build_ssh_bsub_runner(submit_host),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--path-contains")
    parser.add_argument("--zarr-use", default="analysis")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--detection-set-id", required=True)
    parser.add_argument("--detection-run-id", required=True)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--include-postprocess", action="store_true")
    parser.add_argument("--include-crop", action="store_true")
    parser.add_argument(
        "--registered-gate-requirement",
        choices=tuple(sorted(REGISTERED_GATE_REQUIREMENTS)),
        default="off",
    )
    parser.add_argument("--registered-gate-run")
    parser.add_argument(
        "--selection-policy-id",
        choices=("manual_review_only_v1", "corroborated_acquisition_v1"),
        default="manual_review_only_v1",
    )
    parser.add_argument(
        "--submit-host",
        default=os.environ.get("PALETTE_LSF_SUBMIT_HOST", "login1-citrus-poller"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry_path = (
        args.registry
        if args.registry is not None
        else RegistryPaths.from_env(Path.cwd()).path
    )
    plan = build_plan(
        registry_path=registry_path,
        repo=args.repo,
        run_root=args.run_root,
        run_label=args.run_label,
        detection_set_id=args.detection_set_id,
        detection_run_id=args.detection_run_id,
        recording_ids=args.recording_id,
        path_contains=args.path_contains,
        zarr_use=args.zarr_use,
        limit=args.limit,
        max_concurrent=args.max_concurrent,
        include_postprocess=args.include_postprocess,
        include_crop=args.include_crop,
        registered_gate_requirement=args.registered_gate_requirement,
        registered_gate_run=args.registered_gate_run,
        selection_policy_id=args.selection_policy_id,
    )
    result = (
        apply_plan(plan, submit_host=args.submit_host)
        if args.apply
        else materialize_plan_bundle(plan)
    )
    summary = {
        "status": "submitted" if args.apply else "dry_run",
        "plan_path": str(plan.run_root / "plan.json"),
        "lsf_plan_path": str(plan.run_root / "lsf_plan.json"),
        "target_count": len(plan.targets),
        "job_count": len(plan.lsf_workflow.jobs),
        "array_task_count": len(
            plan.lsf_workflow.jobs[0].execution_group.tasks  # type: ignore[union-attr]
        ),
        "max_concurrent": plan.max_concurrent,
        "detect_run": plan.detect_run,
        "quality_run": plan.quality_run,
        "refined_run": plan.refined_run,
        "crop_run": plan.crop_run,
        "registered_dish_geometry": {
            "gate_requirement": plan.registered_gate_requirement,
            "gate_run": plan.registered_gate_run,
            "selection_policy_id": plan.selection_policy_id,
        },
        "model": {
            "set_id": plan.model.set_id,
            "run_id": plan.model.run_id,
            "sha256": plan.model.sha256,
        },
        "result": result if args.apply else None,
    }
    print(json.dumps(summary, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORITATIVE_FULL_FRAME_ROLE",
    "PLAN_SCHEMA",
    "RegistryWholeVideoTarget",
    "WholeVideoDetectionCohortPlan",
    "apply_plan",
    "build_plan",
    "discover_registry_whole_video_targets",
    "main",
    "materialize_plan_bundle",
    "resolve_detection_model_for_targets",
]
