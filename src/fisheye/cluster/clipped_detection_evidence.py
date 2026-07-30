"""Composable strict clip-evidence and binding publication fragments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fisheye.cluster.clipped_lsf import (
    build_execution_task,
    build_job,
    build_task_group_job,
)
from fisheye.cluster.clipped_storage_finalization import (
    ClippedStorageFinalizationInputs,
    ClippedStorageFinalizationModule,
    StrictClipRefinedDetectionInput,
    build_clipped_storage_finalization_fragment,
)
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfExecutionMode,
    LsfResources,
    LsfWorkflowFragment,
)


@dataclass(frozen=True)
class ClipDetectionEvidenceInput:
    clip_index: int
    clip_id: str
    source_detect_group_path: str
    source_refined_group_path: str
    canonical_run_id: str
    refined_run_id: str

    def __post_init__(self) -> None:
        if type(self.clip_index) is not int or self.clip_index < 0:
            raise ValueError("clip_index must be a nonnegative exact integer.")
        for name in ("clip_id", "canonical_run_id", "refined_run_id"):
            value = str(getattr(self, name)).strip()
            if not value or "/" in value:
                raise ValueError(f"{name} must be one path-safe component.")
        for name in ("source_detect_group_path", "source_refined_group_path"):
            value = str(getattr(self, name)).strip().strip("/")
            if not value or ".." in Path(value).parts:
                raise ValueError(f"{name} must be one safe archive-relative path.")


@dataclass(frozen=True)
class ClippedDetectionEvidenceInputs:
    workflow_id: str
    family: str
    target_id: str
    analysis_zarr: Path
    recording_canonical_archive: Path
    recording_canonical_run_id: str
    recording_identity: str
    detection_plan_path: Path
    collection_id: str
    recording_dir: Path
    bundle_root: Path
    clips: tuple[ClipDetectionEvidenceInput, ...]
    repo: Path
    run_root: Path
    max_concurrent: int = 4
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.clips, key=lambda item: item.clip_index))
        if not ordered or tuple(item.clip_index for item in ordered) != tuple(
            range(len(ordered))
        ):
            raise ValueError("Clip evidence inputs must cover [0, clip_count).")
        if len({item.clip_id for item in ordered}) != len(ordered):
            raise ValueError("Clip evidence input ids must be unique.")
        if type(self.max_concurrent) is not int or self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be a positive exact integer.")
        resolved_bundle = self.bundle_root.expanduser().resolve()
        if ".palette_benchmarks" not in resolved_bundle.parts and not str(
            resolved_bundle
        ).startswith("/tmp/"):
            raise ValueError(
                "Strict clip evidence must remain below /tmp or .palette_benchmarks."
            )


@dataclass(frozen=True)
class ClippedDetectionEvidenceOutputs:
    clips: tuple[StrictClipRefinedDetectionInput, ...]
    receipt_paths: tuple[Path, ...]
    clipped_binding_path: Path
    clipped_binding_receipt_path: Path
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "clips": [
                {
                    "clip_index": item.clip_index,
                    "clip_id": item.clip_id,
                    "archive": str(item.archive),
                    "run_id": item.run_id,
                }
                for item in self.clips
            ],
            "receipt_paths": [str(path) for path in self.receipt_paths],
            "clipped_binding_path": str(self.clipped_binding_path),
            "clipped_binding_receipt_path": str(self.clipped_binding_receipt_path),
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
            "selector_eligible": False,
            "registry_updated": False,
        }


@dataclass(frozen=True)
class ClippedDetectionEvidenceModule:
    fragment: LsfWorkflowFragment
    outputs: ClippedDetectionEvidenceOutputs


@dataclass(frozen=True)
class ClippedDetectionStorageModules:
    evidence: ClippedDetectionEvidenceModule
    storage: ClippedStorageFinalizationModule


def _clip_root(bundle_root: Path, clip: ClipDetectionEvidenceInput) -> Path:
    return bundle_root / f"clip_{clip.clip_index:06d}_{clip.clip_id}"


def build_clipped_detection_evidence_fragment(
    inputs: ClippedDetectionEvidenceInputs,
) -> ClippedDetectionEvidenceModule:
    """Publish strict clip pairs, then derive the exact recording binding."""

    target = safe_component(inputs.target_id, default="target", max_length=56)
    ordered = tuple(sorted(inputs.clips, key=lambda item: item.clip_index))
    tasks = []
    receipts: list[Path] = []
    refined_inputs: list[StrictClipRefinedDetectionInput] = []
    for clip in ordered:
        clip_root = _clip_root(inputs.bundle_root, clip)
        receipt = clip_root / "strict_detection_evidence_receipt.json"
        receipts.append(receipt)
        refined_inputs.append(
            StrictClipRefinedDetectionInput(
                clip_index=clip.clip_index,
                clip_id=clip.clip_id,
                archive=clip_root / "refined.zarr",
                run_id=clip.refined_run_id,
            )
        )
        tasks.append(
            build_execution_task(
                run_root=inputs.run_root,
                task_key=f"strict_detection_evidence:{target}:{clip.clip_id}",
                stage="strict_clip_detection_evidence",
                command=(
                    "scripts/py",
                    "-m",
                    "fisheye.utils.publish_strict_clip_detection_evidence",
                    "--analysis-zarr",
                    str(inputs.analysis_zarr),
                    "--source-detect-group",
                    clip.source_detect_group_path,
                    "--source-refined-group",
                    clip.source_refined_group_path,
                    "--recording-canonical-archive",
                    str(inputs.recording_canonical_archive),
                    "--recording-canonical-run",
                    inputs.recording_canonical_run_id,
                    "--recording-identity",
                    inputs.recording_identity,
                    "--clip-id",
                    clip.clip_id,
                    "--clip-index",
                    str(clip.clip_index),
                    "--output-root",
                    str(inputs.bundle_root),
                    "--canonical-run",
                    clip.canonical_run_id,
                    "--refined-run",
                    clip.refined_run_id,
                ),
                expected_outputs=(
                    clip_root / "canonical.zarr" / "zarr.json",
                    clip_root / "refined.zarr" / "zarr.json",
                    receipt,
                ),
                array_indexed=True,
            )
        )
    array_key = f"strict_detection_evidence_array:{target}"
    array_job = build_task_group_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=array_key,
        stage="strict_clip_detection_evidence",
        tasks=tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=min(inputs.max_concurrent, len(tasks)),
        resources=LsfResources(
            queue="short",
            ncores=4,
            mem_gb=32,
            walltime="2:00",
            span_hosts=1,
        ),
        upstream=inputs.upstream_job_keys,
    )

    binding_path = inputs.bundle_root / "clipped_refined_detection_binding.json"
    binding_command: list[str] = [
        "scripts/py",
        "-m",
        "fisheye.utils.build_clipped_refined_detection_binding",
        "--analysis-zarr",
        str(inputs.analysis_zarr),
        "--detection-plan",
        str(inputs.detection_plan_path),
        "--collection-id",
        inputs.collection_id,
        "--recording-frame-index",
        str(inputs.recording_dir / "recording_frame_index.parquet"),
        "--recording-clip-index",
        str(inputs.recording_dir / "recording_clip_index.json"),
        "--output",
        str(binding_path),
    ]
    for receipt in receipts:
        binding_command.extend(("--strict-evidence-receipt", str(receipt)))
    binding_key = f"strict_detection_binding:{target}"
    binding_job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=binding_key,
        stage="strict_clipped_detection_binding",
        command=tuple(binding_command),
        resources=LsfResources(
            queue="short",
            ncores=2,
            mem_gb=16,
            walltime="1:00",
            span_hosts=1,
        ),
        upstream=(array_key,),
        expected_outputs=(
            binding_path,
            binding_path.with_suffix(".receipt.json"),
        ),
    )
    artifact_key = f"strict_clipped_detection_evidence:{target}"
    outputs = ClippedDetectionEvidenceOutputs(
        clips=tuple(refined_inputs),
        receipt_paths=tuple(receipts),
        clipped_binding_path=binding_path,
        clipped_binding_receipt_path=binding_path.with_suffix(".receipt.json"),
        terminal_job_key=binding_key,
        artifact_key=artifact_key,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"strict_clipped_detection_evidence:{target}",
        jobs=(array_job, binding_job),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "strict_clipped_detection_evidence",
            "target_id": inputs.target_id,
            "clip_count": len(ordered),
            "manual_rows_allowed": False,
            "manual_edit_path": "recording_delta_then_compaction",
            "identity_allocation": "recording_canonical_source_row_position_v1",
            "physical_layout_source": "shared_versioned_byte_planners",
            "selector_activation": "none_direct_path_only",
            "registry_update": False,
            "outputs": outputs.to_json(),
        },
    )
    return ClippedDetectionEvidenceModule(fragment=fragment, outputs=outputs)


def build_clipped_detection_storage_fragments(
    evidence_inputs: ClippedDetectionEvidenceInputs,
    storage_inputs: ClippedStorageFinalizationInputs,
) -> ClippedDetectionStorageModules:
    """Freeze evidence -> binding -> recording refined/crop dependency edges."""

    evidence = build_clipped_detection_evidence_fragment(evidence_inputs)
    if evidence_inputs.target_id != storage_inputs.target_id:
        raise ValueError(
            "Evidence and storage finalization target different recordings."
        )
    storage = build_clipped_storage_finalization_fragment(
        replace(
            storage_inputs,
            clips=evidence.outputs.clips,
            clipped_binding_path=evidence.outputs.clipped_binding_path,
            upstream_job_keys=tuple(
                dict.fromkeys(
                    (
                        *storage_inputs.upstream_job_keys,
                        evidence.outputs.terminal_job_key,
                    )
                )
            ),
            required_artifacts=tuple(
                dict.fromkeys(
                    (
                        *storage_inputs.required_artifacts,
                        evidence.outputs.artifact_key,
                    )
                )
            ),
        )
    )
    return ClippedDetectionStorageModules(evidence=evidence, storage=storage)


__all__ = [
    "ClipDetectionEvidenceInput",
    "ClippedDetectionEvidenceInputs",
    "ClippedDetectionEvidenceModule",
    "ClippedDetectionEvidenceOutputs",
    "ClippedDetectionStorageModules",
    "build_clipped_detection_evidence_fragment",
    "build_clipped_detection_storage_fragments",
]
