"""Composable whole-acquisition crop-snapshot publication fragment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fisheye.cluster.clipped_lsf import build_job, chain_commands
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import LsfResources, LsfWorkflowFragment
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN, RUNTIME_USER_TOKEN


@dataclass(frozen=True)
class CropSnapshotFragmentInputs:
    workflow_id: str
    family: str
    target_id: str
    analysis_zarr: Path
    repo: Path
    run_root: Path
    run_id: str
    purpose: str
    roi_width: int | None
    roi_height: int | None
    camera_id: str
    source_refined_run: str | None = None
    registered_gate_requirement: str = "off"
    registered_gate_run: str | None = None
    geometry_origin_provider_run: str | None = None
    roi_size_from_geometry_origin_provider: bool = False
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    copy_backend: str = "python"

    def __post_init__(self) -> None:
        if self.roi_size_from_geometry_origin_provider:
            if not str(self.geometry_origin_provider_run or "").strip():
                raise ValueError(
                    "Provider-derived ROI size requires geometry_origin_provider_run."
                )
        else:
            for name in ("roi_width", "roi_height"):
                if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be a positive exact integer.")
        if self.copy_backend not in {"python", "rsync"}:
            raise ValueError("copy_backend must be 'python' or 'rsync'.")
        if self.registered_gate_requirement not in {
            "off",
            "if_available",
            "required",
            "from_source",
        }:
            raise ValueError(
                "registered_gate_requirement must be off, if_available, required, "
                "or from_source."
            )
        if (
            self.registered_gate_requirement != "off"
            and not str(self.source_refined_run or "").strip()
        ):
            raise ValueError(
                "Configured registered geometry requires an exact finalized refined run."
            )
        if (
            self.registered_gate_requirement == "required"
            and not str(self.registered_gate_run or "").strip()
        ):
            raise ValueError("Required registered geometry needs one exact gate run.")


@dataclass(frozen=True)
class CropSnapshotFragmentOutputs:
    target_id: str
    run_id: str
    group_path: str
    publication_receipt_path: Path
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "run_id": self.run_id,
            "group_path": self.group_path,
            "publication_receipt_path": str(self.publication_receipt_path),
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
            "logical_schema_version": 1,
            "run_manifest_schema_version": 2,
            "selector_eligible": False,
            "registry_updated": False,
        }


@dataclass(frozen=True)
class CropSnapshotWorkflowModule:
    fragment: LsfWorkflowFragment
    outputs: CropSnapshotFragmentOutputs


def build_crop_snapshot_fragment(
    inputs: CropSnapshotFragmentInputs,
) -> CropSnapshotWorkflowModule:
    """Plan the same strict crop publication for clipped or whole workflows."""

    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    job_key = f"crop_snapshot_publish:{target_safe}"
    receipt = inputs.run_root / "crop_snapshot" / f"{target_safe}.publication.json"
    scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/crop_geometry_candidate"
    )
    command = [
        "scripts/py",
        "-m",
        "fisheye.utils.publish_crop_geometry_candidate",
        "--analysis-zarr",
        str(inputs.analysis_zarr),
        "--run-id",
        inputs.run_id,
        "--purpose",
        inputs.purpose,
        "--camera-id",
        inputs.camera_id,
        "--scratch-root",
        scratch,
        "--copy-backend",
        inputs.copy_backend,
        "--result-json",
        str(receipt),
    ]
    if inputs.roi_size_from_geometry_origin_provider:
        command.append("--roi-size-from-geometry-origin-provider")
    else:
        assert inputs.roi_width is not None and inputs.roi_height is not None
        command.extend(
            (
                "--roi-width",
                str(inputs.roi_width),
                "--roi-height",
                str(inputs.roi_height),
            )
        )
    if inputs.source_refined_run is not None:
        command.extend(("--source-refined-run", inputs.source_refined_run))
    if inputs.geometry_origin_provider_run is not None:
        command.extend(
            (
                "--geometry-origin-provider-run",
                inputs.geometry_origin_provider_run,
            )
        )
    command.extend(
        ("--registered-gate-requirement", inputs.registered_gate_requirement)
    )
    if inputs.registered_gate_run is not None:
        command.extend(("--registered-gate-run", inputs.registered_gate_run))
    resources = LsfResources(
        queue="short",
        ncores=4,
        mem_gb=32,
        walltime="1:00",
        span_hosts=1,
    )
    job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=job_key,
        stage="crop_snapshot_publish",
        command=chain_commands((("mkdir", "-p", scratch), tuple(command))),
        resources=resources,
        upstream=inputs.upstream_job_keys,
        expected_outputs=(
            inputs.analysis_zarr / "crop_runs" / inputs.run_id / "zarr.json",
            receipt,
        ),
        cleanup_paths=(scratch,),
    )
    artifact_key = f"crop_snapshot:{target_safe}"
    outputs = CropSnapshotFragmentOutputs(
        target_id=inputs.target_id,
        run_id=inputs.run_id,
        group_path=f"crop_runs/{inputs.run_id}",
        publication_receipt_path=receipt,
        terminal_job_key=job_key,
        artifact_key=artifact_key,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"crop_snapshot:{target_safe}",
        jobs=(job,),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "crop_snapshot",
            "target_id": inputs.target_id,
            "canonical_namespace": "crop_runs",
            "lineage_profile": "full_acquisition",
            "compute_partition_independent": True,
            "selector_activation": "deferred",
            "registry_update": False,
            "source_refined_run": inputs.source_refined_run,
            "registered_gate_requirement": inputs.registered_gate_requirement,
            "registered_gate_run": inputs.registered_gate_run,
            "geometry_origin_provider_run": inputs.geometry_origin_provider_run,
            "roi_size_from_geometry_origin_provider": (
                inputs.roi_size_from_geometry_origin_provider
            ),
            "outputs": outputs.to_json(),
        },
    )
    return CropSnapshotWorkflowModule(fragment=fragment, outputs=outputs)


__all__ = [
    "CropSnapshotFragmentInputs",
    "CropSnapshotFragmentOutputs",
    "CropSnapshotWorkflowModule",
    "build_crop_snapshot_fragment",
]
