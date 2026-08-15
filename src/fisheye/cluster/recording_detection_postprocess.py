"""Shared canonical detect-quality and refinement DAG for both video layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fisheye.cluster.clipped_lsf import build_job, chain_commands
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import LsfResources, LsfWorkflowFragment
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN
from fisheye.cluster.recording_layout import RecordingTarget

REGISTERED_GATE_REQUIREMENTS = frozenset({"off", "if_available", "required"})


@dataclass(frozen=True)
class RecordingDetectionPostprocessInputs:
    workflow_id: str
    family: str
    target: RecordingTarget
    repo: Path
    run_root: Path
    source_detect_run: str
    quality_run: str
    refined_run: str
    canonicalize_legacy_source: bool = False
    canonical_source_run: str | None = None
    require_active_canonical_source: bool = False
    registered_gate_requirement: str = "off"
    registered_gate_run: str | None = None
    selection_policy_id: str = "manual_review_only_v1"
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    quality_workers: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.target, RecordingTarget):
            raise TypeError("Detection postprocess target must be a RecordingTarget.")
        for name in (
            "workflow_id",
            "family",
            "source_detect_run",
            "quality_run",
            "refined_run",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Detection postprocess {name} cannot be empty.")
        if type(self.canonicalize_legacy_source) is not bool:
            raise TypeError("canonicalize_legacy_source must be an exact bool.")
        if type(self.require_active_canonical_source) is not bool:
            raise TypeError("require_active_canonical_source must be an exact bool.")
        canonical_source_run = str(self.canonical_source_run or "").strip() or None
        object.__setattr__(self, "canonical_source_run", canonical_source_run)
        if self.canonicalize_legacy_source and canonical_source_run is None:
            raise ValueError(
                "Legacy detection canonicalization requires a successor run id."
            )
        if not self.canonicalize_legacy_source and canonical_source_run is not None:
            raise ValueError(
                "canonical_source_run is only valid when canonicalize_legacy_source "
                "is true."
            )
        if canonical_source_run == self.source_detect_run:
            raise ValueError(
                "Canonical detection successor must differ from its source run."
            )
        if self.canonicalize_legacy_source and self.require_active_canonical_source:
            raise ValueError(
                "Legacy successor compatibility cannot claim active canonical authority."
            )
        requirement = str(self.registered_gate_requirement).strip()
        if requirement not in REGISTERED_GATE_REQUIREMENTS:
            raise ValueError(
                "registered_gate_requirement must be off, if_available, or required."
            )
        object.__setattr__(self, "registered_gate_requirement", requirement)
        if (
            requirement == "required"
            and not str(self.registered_gate_run or "").strip()
        ):
            raise ValueError("Required registered geometry needs an exact gate run.")
        if self.selection_policy_id not in {
            "manual_review_only_v1",
            "corroborated_acquisition_v1",
        }:
            raise ValueError("Unsupported registered geometry selection policy id.")
        if int(self.quality_workers) <= 0:
            raise ValueError("quality_workers must be positive.")


@dataclass(frozen=True)
class RecordingDetectionPostprocessOutputs:
    target_id: str
    input_detect_run: str
    input_detect_group_path: str
    source_detect_run: str
    source_detect_group_path: str
    canonical_successor_receipt: str | None
    require_active_canonical_source: bool
    quality_run: str
    quality_group_path: str
    working_refined_run: str
    working_refined_group_path: str
    refined_run: str
    refined_group_path: str
    registered_gate_requirement: str
    registered_gate_run: str | None
    selection_policy_id: str
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "input_detect_run": self.input_detect_run,
            "input_detect_group_path": self.input_detect_group_path,
            "source_detect_run": self.source_detect_run,
            "source_detect_group_path": self.source_detect_group_path,
            "canonical_successor_receipt": self.canonical_successor_receipt,
            "require_active_canonical_source": (
                self.require_active_canonical_source
            ),
            "quality_run": self.quality_run,
            "quality_group_path": self.quality_group_path,
            "working_refined_run": self.working_refined_run,
            "working_refined_group_path": self.working_refined_group_path,
            "refined_run": self.refined_run,
            "refined_group_path": self.refined_group_path,
            "registered_gate_requirement": self.registered_gate_requirement,
            "registered_gate_run": self.registered_gate_run,
            "selection_policy_id": self.selection_policy_id,
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
            "row_identity": "instance_key",
            "publication_partition": "complete_recording_snapshot",
        }


@dataclass(frozen=True)
class RecordingDetectionPostprocessModule:
    fragment: LsfWorkflowFragment
    outputs: RecordingDetectionPostprocessOutputs


def build_recording_detection_postprocess_fragment(
    inputs: RecordingDetectionPostprocessInputs,
) -> RecordingDetectionPostprocessModule:
    """Plan canonical raw -> quality -> optional gate join -> refined snapshot."""

    safe = safe_component(inputs.target.target_id, default="target", max_length=56)
    input_source_group = f"detect_runs/{inputs.source_detect_run}"
    effective_source_run = (
        inputs.canonical_source_run
        if inputs.canonicalize_legacy_source
        else inputs.source_detect_run
    )
    assert effective_source_run is not None
    source_group = f"detect_runs/{effective_source_run}"
    quality_group = f"detect_quality_runs/{inputs.quality_run}"
    working_refined_run = safe_component(
        f"{inputs.refined_run}__working",
        default="refined_detect_working",
        max_length=120,
    )
    working_refined_group = f"refined_detect_runs/{working_refined_run}"
    refined_group = f"refined_detect_runs/{inputs.refined_run}"
    work_dir = (
        f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}/"
        "recording_detect_quality"
    )
    jobs = []
    quality_upstream = inputs.upstream_job_keys
    canonical_successor_receipt: Path | None = None
    if inputs.canonicalize_legacy_source:
        canonical_key = f"recording_detect_canonicalize:{safe}"
        canonical_scratch = (
            f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}/"
            "recording_detect_canonicalize"
        )
        canonical_successor_receipt = (
            inputs.run_root / "canonical_detection" / f"{safe}.json"
        )
        canonicalize = build_job(
            workflow_id=inputs.workflow_id,
            family=inputs.family,
            repo=inputs.repo,
            run_root=inputs.run_root,
            job_key=canonical_key,
            stage="detect_canonicalize",
            command=chain_commands(
                (
                    ("mkdir", "-p", canonical_scratch),
                    (
                        "scripts/py",
                        "-m",
                        "fisheye.utils.publish_canonical_detection_successor",
                        "--analysis-zarr",
                        str(inputs.target.analysis_zarr),
                        "--source-detect-group",
                        input_source_group,
                        "--recording-identity",
                        inputs.target.recording_id,
                        "--successor-run",
                        effective_source_run,
                        "--scratch-root",
                        canonical_scratch,
                        "--result-json",
                        str(canonical_successor_receipt),
                        "--apply",
                    ),
                )
            ),
            resources=LsfResources(queue="short", ncores=4, mem_gb=32, walltime="1:00"),
            upstream=inputs.upstream_job_keys,
            expected_outputs=(
                inputs.target.analysis_zarr / source_group / "zarr.json",
                canonical_successor_receipt,
            ),
            cleanup_paths=(canonical_scratch,),
        )
        jobs.append(canonicalize)
        quality_upstream = (canonical_key,)

    quality_key = f"recording_detect_quality:{safe}"
    quality = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=quality_key,
        stage="detect_quality",
        command=(
            "scripts/py",
            "-m",
            "fisheye.refinement.detect_quality_collection",
            str(inputs.target.analysis_zarr),
            "--source-group",
            source_group,
            "--output-run",
            inputs.quality_run,
            "--expected-subject-count",
            str(inputs.target.expected_subject_count),
            "--threshold-mode",
            "scaled",
            "--jump-threshold",
            "100.0",
            "--threshold-reference-width",
            "640.0",
            "--blip-gap-threshold",
            "10",
            "--shard-rows",
            "131072",
            "--row-chunk-rows",
            "16384",
            "--frame-chunk-rows",
            "16384",
            "--workers",
            str(int(inputs.quality_workers)),
            "--work-dir",
            work_dir,
            "--apply",
            "--json",
            *(
                ("--require-active-canonical-source",)
                if inputs.require_active_canonical_source
                else ()
            ),
        ),
        resources=LsfResources(queue="short", ncores=4, mem_gb=32, walltime="1:00"),
        upstream=quality_upstream,
        expected_outputs=(inputs.target.analysis_zarr / quality_group / "zarr.json",),
        cleanup_paths=(work_dir,),
    )
    refine_command = [
        "scripts/py",
        "-m",
        "fisheye.refinement.refine_detect",
        str(inputs.target.analysis_zarr),
        "--detect-run",
        effective_source_run,
        "--quality-group-path",
        quality_group,
        "--config",
        str(inputs.repo / "configs" / "fisheye" / "default.yaml"),
        "--run-name",
        working_refined_run,
        "--per-frame-top-k",
        "1",
        "--registered-gate-requirement",
        inputs.registered_gate_requirement,
        "--selector-ineligible",
        "--skip-completion-status",
    ]
    if inputs.registered_gate_run is not None:
        refine_command.extend(("--registered-gate-run", inputs.registered_gate_run))
    if inputs.require_active_canonical_source:
        refine_command.append("--require-active-canonical-source")
    validate_command = (
        "scripts/py",
        "-m",
        "fisheye.utils.validate_refined_detect_run",
        str(inputs.target.analysis_zarr),
        "--target-group-path",
        working_refined_group,
    )
    finalize_scratch = (
        f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}/"
        "recording_refined_finalize"
    )
    finalization_receipt = (
        inputs.run_root / "finalization" / f"{safe}.refined_detection.json"
    )
    finalize_command = [
        "scripts/py",
        "-m",
        "fisheye.utils.finalize_recording_refined_detection_v1",
        "--analysis-zarr",
        str(inputs.target.analysis_zarr),
        "--canonical-detect-run",
        effective_source_run,
        "--working-refined-run",
        working_refined_run,
        "--output-run",
        inputs.refined_run,
        "--recording-identity",
        inputs.target.recording_id,
        "--registered-gate-requirement",
        inputs.registered_gate_requirement,
        "--selection-policy-id",
        inputs.selection_policy_id,
        "--scratch-root",
        finalize_scratch,
        "--result-json",
        str(finalization_receipt),
    ]
    if inputs.registered_gate_run is not None:
        finalize_command.extend(("--registered-gate-run", inputs.registered_gate_run))
    if inputs.require_active_canonical_source:
        finalize_command.append("--require-active-canonical-source")
    refine_key = f"recording_detect_refine:{safe}"
    refine = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=refine_key,
        stage="detect_refine",
        command=chain_commands(
            (
                ("mkdir", "-p", finalize_scratch),
                tuple(refine_command),
                validate_command,
                tuple(finalize_command),
            )
        ),
        resources=LsfResources(queue="short", ncores=4, mem_gb=32, walltime="1:00"),
        upstream=(quality_key,),
        expected_outputs=(
            inputs.target.analysis_zarr / working_refined_group / "zarr.json",
            inputs.target.analysis_zarr / refined_group / "zarr.json",
            finalization_receipt,
        ),
        cleanup_paths=(finalize_scratch,),
    )
    artifact_key = f"canonical_refined_detection:{safe}"
    jobs.extend((quality, refine))
    outputs = RecordingDetectionPostprocessOutputs(
        target_id=inputs.target.target_id,
        input_detect_run=inputs.source_detect_run,
        input_detect_group_path=input_source_group,
        source_detect_run=effective_source_run,
        source_detect_group_path=source_group,
        canonical_successor_receipt=(
            str(canonical_successor_receipt)
            if canonical_successor_receipt is not None
            else None
        ),
        require_active_canonical_source=inputs.require_active_canonical_source,
        quality_run=inputs.quality_run,
        quality_group_path=quality_group,
        working_refined_run=working_refined_run,
        working_refined_group_path=working_refined_group,
        refined_run=inputs.refined_run,
        refined_group_path=refined_group,
        registered_gate_requirement=inputs.registered_gate_requirement,
        registered_gate_run=inputs.registered_gate_run,
        selection_policy_id=inputs.selection_policy_id,
        terminal_job_key=refine_key,
        artifact_key=artifact_key,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"recording_detection_postprocess:{safe}",
        jobs=tuple(jobs),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "recording_detection_postprocess",
            "recording_layout": inputs.target.layout.value,
            "source_authority": "canonical_recording_detect_run",
            "canonicalize_legacy_source": inputs.canonicalize_legacy_source,
            "refined_authority": "canonical_recording_refined_detect_run",
            "refined_finalization": "immutable_refined_v1_snapshot",
            "registered_gate_requirement": inputs.registered_gate_requirement,
            "registered_gate_run": inputs.registered_gate_run,
            "selection_policy_id": inputs.selection_policy_id,
            "raw_detections_preserved": True,
            "legacy_dish_mask_policy_coupled": False,
            "outputs": outputs.to_json(),
        },
    )
    return RecordingDetectionPostprocessModule(fragment=fragment, outputs=outputs)


__all__ = [
    "REGISTERED_GATE_REQUIREMENTS",
    "RecordingDetectionPostprocessInputs",
    "RecordingDetectionPostprocessModule",
    "RecordingDetectionPostprocessOutputs",
    "build_recording_detection_postprocess_fragment",
]
