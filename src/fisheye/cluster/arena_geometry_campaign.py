"""Plan and submit recording-level arena-geometry review campaigns.

This is the pre-review workflow only.  Each target publishes its immutable
acquisition candidate and independently generates a blind keyframe-only
Palette fit/reveal package.  The campaign stops before reviewed-candidate
publication, comparison, operational selection, or detection gating.  A final
serialized job refreshes registry projections from the immutable Zarr results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.arena_geometry_review import (
    ArenaGeometryProbeSource,
    ArenaGeometryReviewArrayWorkflowModule,
    ArenaGeometryReviewFragmentInputs,
    build_arena_geometry_review_array_fragment,
    compose_arena_geometry_workflow,
    validate_recording_level_probe_source,
)
from fisheye.cluster.clipped_inference import DEFAULT_REGISTRY
from fisheye.cluster.keypoints.common import (
    safe_component,
    validate_registered_analysis_zarr,
)
from fisheye.cluster.lsf import (
    CommandRunner,
    LsfResources,
    LsfWorkflow,
    build_ssh_bsub_runner,
    submit_lsf_workflow,
    write_json_snapshot,
)

TARGET_MANIFEST_SCHEMA = "palette.arena_geometry_review_targets.v1"
PLAN_SCHEMA = "palette.arena_geometry_review_campaign_plan.v1"
DEFAULT_SUBMIT_HOST = "login1-citrus-poller"


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _contained_file(path: Path, recording_dir: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(recording_dir)
    except ValueError as exc:
        raise ValueError(f"{label} must belong to the recording directory.") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


@dataclass(frozen=True)
class ArenaGeometryTarget:
    target_id: str
    recording_id: str
    recording_dir: Path
    analysis_zarr: Path
    video_path: Path | None = None
    summary_path: Path | None = None
    keyframe_path: Path | None = None
    recovery_receipt_path: Path | None = None
    acquisition_observation_path: Path | None = None
    geometry_source: str = "recovery-receipt"
    geometry_camera_serial: str | None = None
    geometry_arena_id: str | None = None
    citrus_h5_path: Path | None = None

    def __post_init__(self) -> None:
        recording = self.recording_dir.expanduser().resolve()
        if not recording.is_dir():
            raise FileNotFoundError(f"Recording directory not found: {recording}")
        target_id = safe_component(self.target_id, default="target", max_length=80)
        recording_id = str(self.recording_id).strip()
        if not recording_id:
            raise ValueError(f"Target {target_id!r} has no recording_id.")
        analysis = self.analysis_zarr.expanduser().resolve()
        try:
            analysis.relative_to(recording)
        except ValueError as exc:
            raise ValueError("Analysis Zarr must belong to the recording.") from exc
        if not (analysis / "zarr.json").is_file():
            raise FileNotFoundError(f"Analysis target is not Zarr v3: {analysis}")
        observation = self.acquisition_observation_path
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "recording_id", recording_id)
        object.__setattr__(self, "recording_dir", recording)
        object.__setattr__(self, "analysis_zarr", analysis)
        probe_source = ArenaGeometryProbeSource(
            video_path=self.video_path,
            summary_path=self.summary_path,
            keyframe_path=self.keyframe_path,
            recording_dir=(
                recording
                if all(
                    value is None
                    for value in (
                        self.video_path,
                        self.summary_path,
                        self.keyframe_path,
                    )
                )
                else None
            ),
            acquisition_observation_path=self.acquisition_observation_path,
        )
        validate_recording_level_probe_source(recording, probe_source)
        object.__setattr__(self, "video_path", probe_source.video_path)
        object.__setattr__(self, "summary_path", probe_source.summary_path)
        object.__setattr__(self, "keyframe_path", probe_source.keyframe_path)
        geometry_source = str(self.geometry_source).strip()
        if geometry_source not in {
            "producer-folder",
            "citrus-h5",
            "recovery-receipt",
        }:
            raise ValueError(f"Unsupported geometry source: {geometry_source!r}.")
        receipt = None
        if geometry_source == "recovery-receipt":
            if self.recovery_receipt_path is None:
                raise ValueError("Recovery geometry requires recovery_receipt.")
            receipt = _contained_file(
                self.recovery_receipt_path,
                recording,
                label="geometry recovery receipt",
            )
        elif self.recovery_receipt_path is not None:
            raise ValueError(
                "Producer-native geometry must not declare recovery_receipt."
            )
        camera_serial = (
            str(self.geometry_camera_serial).strip()
            if self.geometry_camera_serial is not None
            else ""
        )
        arena_id = (
            str(self.geometry_arena_id).strip()
            if self.geometry_arena_id is not None
            else ""
        )
        if geometry_source != "recovery-receipt" and not (camera_serial and arena_id):
            raise ValueError(
                "Producer-native geometry requires geometry_camera_serial and "
                "geometry_arena_id."
            )
        citrus_h5 = None
        if geometry_source == "citrus-h5":
            if self.citrus_h5_path is None:
                raise ValueError("citrus-h5 geometry requires citrus_h5.")
            citrus_h5 = _contained_file(
                self.citrus_h5_path,
                recording,
                label="recording-bound Citrus H5",
            )
        elif self.citrus_h5_path is not None:
            raise ValueError("citrus_h5 is only valid for citrus-h5 geometry.")
        object.__setattr__(self, "recovery_receipt_path", receipt)
        object.__setattr__(self, "geometry_source", geometry_source)
        object.__setattr__(self, "geometry_camera_serial", camera_serial or None)
        object.__setattr__(self, "geometry_arena_id", arena_id or None)
        object.__setattr__(self, "citrus_h5_path", citrus_h5)
        if observation is not None:
            object.__setattr__(
                self,
                "acquisition_observation_path",
                _contained_file(
                    observation,
                    recording,
                    label="acquisition rim observation",
                ),
            )

    def probe_source(self) -> ArenaGeometryProbeSource:
        return ArenaGeometryProbeSource(
            video_path=self.video_path,
            summary_path=self.summary_path,
            keyframe_path=self.keyframe_path,
            recording_dir=(self.recording_dir if self.video_path is None else None),
            acquisition_observation_path=self.acquisition_observation_path,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "recording_id": self.recording_id,
            "recording_dir": str(self.recording_dir),
            "analysis_zarr": str(self.analysis_zarr),
            "probe_source": self.probe_source().to_json(),
            "video": str(self.video_path) if self.video_path is not None else None,
            "summary": (
                str(self.summary_path) if self.summary_path is not None else None
            ),
            "keyframes": (
                str(self.keyframe_path) if self.keyframe_path is not None else None
            ),
            "geometry_source": self.geometry_source,
            "geometry_camera_serial": self.geometry_camera_serial,
            "geometry_arena_id": self.geometry_arena_id,
            "recovery_receipt": (
                str(self.recovery_receipt_path)
                if self.recovery_receipt_path is not None
                else None
            ),
            "citrus_h5": (
                str(self.citrus_h5_path) if self.citrus_h5_path is not None else None
            ),
            "acquisition_observation": (
                str(self.acquisition_observation_path)
                if self.acquisition_observation_path is not None
                else None
            ),
        }


def load_target_manifest(path: Path) -> tuple[ArenaGeometryTarget, ...]:
    payload = _read_json_object(path.expanduser().resolve())
    if payload.get("schema") != TARGET_MANIFEST_SCHEMA:
        raise ValueError(f"Target manifest schema must be {TARGET_MANIFEST_SCHEMA!r}.")
    rows = payload.get("targets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Target manifest requires a non-empty targets list.")
    targets: list[ArenaGeometryTarget] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Target row {index} is not an object.")
        recording_dir = Path(str(row.get("recording_dir") or ""))
        observation = row.get("acquisition_observation")
        recovery = row.get("recovery_receipt")
        citrus_h5 = row.get("citrus_h5")
        video = row.get("video")
        summary = row.get("summary")
        keyframes = row.get("keyframes")
        targets.append(
            ArenaGeometryTarget(
                target_id=str(row.get("target_id") or recording_dir.name),
                recording_id=str(row.get("recording_id") or ""),
                recording_dir=recording_dir,
                analysis_zarr=Path(str(row.get("analysis_zarr") or "")),
                video_path=Path(str(video)) if video else None,
                summary_path=Path(str(summary)) if summary else None,
                keyframe_path=Path(str(keyframes)) if keyframes else None,
                recovery_receipt_path=(Path(str(recovery)) if recovery else None),
                acquisition_observation_path=(
                    Path(str(observation)) if observation else None
                ),
                geometry_source=str(row.get("geometry_source") or "recovery-receipt"),
                geometry_camera_serial=(
                    str(row.get("geometry_camera_serial"))
                    if row.get("geometry_camera_serial") is not None
                    else None
                ),
                geometry_arena_id=(
                    str(row.get("geometry_arena_id"))
                    if row.get("geometry_arena_id") is not None
                    else None
                ),
                citrus_h5_path=(Path(str(citrus_h5)) if citrus_h5 else None),
            )
        )
    if len({target.target_id for target in targets}) != len(targets):
        raise ValueError("Target ids must be unique.")
    if len({target.analysis_zarr for target in targets}) != len(targets):
        raise ValueError("Analysis Zarr targets must be unique.")
    return tuple(targets)


def _repo_commit(repo: Path) -> str:
    resolved = repo.expanduser().resolve()
    status = subprocess.run(
        ["git", "-C", str(resolved), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        text=True,
        capture_output=True,
    )
    if status.stdout.strip():
        raise ValueError(f"Palette repo must be clean: {resolved}")
    commit = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("Palette repo did not resolve one full commit SHA.")
    return commit


@dataclass(frozen=True)
class ArenaGeometryCampaignPlan:
    run_label: str
    workflow_id: str
    repo: Path
    repo_commit: str
    registry: Path
    run_root: Path
    probe_queue: str
    targets: tuple[ArenaGeometryTarget, ...]
    module: ArenaGeometryReviewArrayWorkflowModule
    workflow: LsfWorkflow

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "run_label": self.run_label,
            "workflow_id": self.workflow_id,
            "repo": str(self.repo),
            "repo_commit": self.repo_commit,
            "registry": str(self.registry),
            "run_root": str(self.run_root),
            "probe_queue": self.probe_queue,
            "target_count": len(self.targets),
            "targets": [
                {**target.to_json(), "outputs": output.to_json()}
                for target, output in zip(
                    self.targets, self.module.outputs, strict=True
                )
            ],
            "execution_mode": "lsf_arrays",
            "human_review_barrier": True,
            "post_review_publication": "not_scheduled",
            "candidate_comparison": "not_scheduled",
            "operational_selection": "not_scheduled",
            "detection_gating": "not_scheduled",
            "registry_update": True,
            "lsf_workflow": self.workflow.to_json(),
        }


def build_plan(
    *,
    targets: Sequence[ArenaGeometryTarget],
    run_label: str,
    repo: Path,
    registry_path: Path,
    run_root: Path,
    acquisition_array_concurrency: int = 8,
    probe_array_concurrency: int = 4,
    probe_queue: str = "gpu_l4",
) -> ArenaGeometryCampaignPlan:
    if not targets:
        raise ValueError("Arena-geometry campaign requires at least one target.")
    label = safe_component(run_label, default="arena_geometry", max_length=80)
    workflow_id = f"arena_geometry_{label}"
    resolved_repo = repo.expanduser().resolve()
    resolved_registry = registry_path.expanduser().resolve()
    resolved_run_root = run_root.expanduser().resolve()
    queue = str(probe_queue).strip()
    if queue not in {"gpu_l4", "gpu_t4"}:
        raise ValueError("Geometry probe queue must be gpu_l4 or gpu_t4.")
    commit = _repo_commit(resolved_repo)
    fragment_inputs: list[ArenaGeometryReviewFragmentInputs] = []
    for target in targets:
        validate_registered_analysis_zarr(
            registry_path=resolved_registry,
            recording_id=target.recording_id,
            analysis_zarr=target.analysis_zarr,
        )
        fragment_inputs.append(
            ArenaGeometryReviewFragmentInputs(
                workflow_id=workflow_id,
                target_id=target.target_id,
                recording_dir=target.recording_dir,
                analysis_zarr=target.analysis_zarr,
                recovery_receipt_path=target.recovery_receipt_path,
                geometry_source=target.geometry_source,
                geometry_camera_serial=target.geometry_camera_serial,
                geometry_arena_id=target.geometry_arena_id,
                citrus_h5_path=target.citrus_h5_path,
                source=target.probe_source(),
                repo=resolved_repo,
                run_root=resolved_run_root,
                registry_path=resolved_registry,
                probe_resources=LsfResources(
                    queue=queue,
                    ncores=8,
                    mem_gb=32,
                    gpus=1,
                    walltime="1:00",
                    span_hosts=1,
                ),
            )
        )
    module = build_arena_geometry_review_array_fragment(
        tuple(fragment_inputs),
        acquisition_max_concurrent=int(acquisition_array_concurrency),
        probe_max_concurrent=int(probe_array_concurrency),
    )
    workflow = compose_arena_geometry_workflow(
        workflow_id=workflow_id,
        modules=(module,),
    )
    return ArenaGeometryCampaignPlan(
        run_label=label,
        workflow_id=workflow_id,
        repo=resolved_repo,
        repo_commit=commit,
        registry=resolved_registry,
        run_root=resolved_run_root,
        probe_queue=queue,
        targets=tuple(targets),
        module=module,
        workflow=workflow,
    )


def materialize_plan_bundle(plan: ArenaGeometryCampaignPlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = _read_json_object(plan_path)
        if existing != payload:
            raise FileExistsError(f"Run root contains a different plan: {plan_path}")
        if (
            not lsf_path.is_file()
            or _read_json_object(lsf_path) != plan.workflow.to_json()
        ):
            raise FileExistsError(f"Run root has mismatched LSF evidence: {lsf_path}")
        return existing
    for name in ("logs", "status", "arena_geometry"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.workflow.to_json())
    return payload


def apply_plan(
    plan: ArenaGeometryCampaignPlan,
    *,
    runner: CommandRunner,
) -> dict[str, Any]:
    submission_path = plan.run_root / "lsf_submission.json"
    if submission_path.exists():
        raise FileExistsError(f"Submission evidence already exists: {submission_path}")
    materialize_plan_bundle(plan)
    return submit_lsf_workflow(
        plan.workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=submission_path,
        runner=runner,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--acquisition-array-concurrency", type=int, default=8)
    parser.add_argument("--probe-array-concurrency", type=int, default=4)
    parser.add_argument(
        "--probe-queue",
        choices=("gpu_l4", "gpu_t4"),
        default="gpu_l4",
    )
    parser.add_argument(
        "--submit-host",
        default=os.environ.get("PALETTE_LSF_SUBMIT_HOST", DEFAULT_SUBMIT_HOST),
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
        acquisition_array_concurrency=args.acquisition_array_concurrency,
        probe_array_concurrency=args.probe_array_concurrency,
        probe_queue=args.probe_queue,
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
        "submission_path": (
            str(plan.run_root / "lsf_submission.json") if args.apply else None
        ),
        "target_count": len(plan.targets),
        "job_count": len(plan.workflow.jobs),
        "execution_mode": "lsf_arrays",
        "probe_queue": plan.probe_queue,
        "human_review_barrier": True,
        "post_review_publication": "not_scheduled",
        "operational_selection": "not_scheduled",
        "detection_gating": "not_scheduled",
        "registry_update": True,
        "result": result if args.apply else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"{summary['status']}: {summary['target_count']} targets, "
            f"{summary['job_count']} jobs; stopped before human review"
        )
        print(f"Plan: {summary['plan_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
