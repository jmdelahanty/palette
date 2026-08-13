"""Composable LSF fragments for recording-level arena-geometry review.

The pre-review fragment publishes immutable acquisition evidence and creates a
blind Palette fit package.  It deliberately stops at a human-review barrier.
Reviewed Palette candidate publication is a separate fragment requiring an
explicit review artifact supplied by a later workflow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    plan_producer_native_acquisition_geometry_candidate,
    plan_recovered_acquisition_geometry_candidate,
    plan_reviewed_palette_geometry_candidate,
)
from fisheye.cluster.clipped_lsf import (
    build_execution_task,
    build_job,
    build_task_group_job,
    chain_commands,
)
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfExecutionMode,
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_JOB_INDEX_TOKEN,
    RUNTIME_USER_TOKEN,
)

FAMILY = "arena_geometry"
REVIEW_PACKAGE_SCHEMA_ID = "palette.diagnostics.recording_dish_rim_probe.review_package"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


@dataclass(frozen=True)
class ArenaGeometryProbeSource:
    """Exact whole-recording source used for the recording-level fit.

    Downstream processing may be clipped or whole-recording.  Geometry remains
    one recording-level artifact in either case; this source is the canonical
    full-frame recording evidence from which it is estimated.
    """

    video_path: Path
    summary_path: Path
    keyframe_path: Path
    acquisition_observation_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "video_path", _require_file(self.video_path, label="source video")
        )
        object.__setattr__(
            self,
            "summary_path",
            _require_file(self.summary_path, label="source summary"),
        )
        object.__setattr__(
            self,
            "keyframe_path",
            _require_file(self.keyframe_path, label="keyframe summary"),
        )
        if self.acquisition_observation_path is not None:
            object.__setattr__(
                self,
                "acquisition_observation_path",
                _require_file(
                    self.acquisition_observation_path,
                    label="acquisition rim observation",
                ),
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "source_kind": "recording_level_whole_video",
            "video_path": str(self.video_path),
            "summary_path": str(self.summary_path),
            "keyframe_path": str(self.keyframe_path),
            "acquisition_observation_path": (
                str(self.acquisition_observation_path)
                if self.acquisition_observation_path is not None
                else None
            ),
        }


def validate_recording_level_probe_source(
    recording_dir: Path,
    source: ArenaGeometryProbeSource,
) -> None:
    """Require one native recording video and its exact organized sidecars."""

    recording = recording_dir.expanduser().resolve()
    source_paths = {
        "source video": source.video_path,
        "source summary": source.summary_path,
        "keyframe summary": source.keyframe_path,
    }
    if source.acquisition_observation_path is not None:
        source_paths["acquisition rim observation"] = (
            source.acquisition_observation_path
        )
    for label, path in source_paths.items():
        try:
            path.relative_to(recording)
        except ValueError as exc:
            raise ValueError(
                f"Arena-geometry {label} must belong to the target recording."
            ) from exc

    camera_dir = (recording / "cams").resolve()
    if any(
        path.parent != camera_dir
        for path in (source.video_path, source.summary_path, source.keyframe_path)
    ):
        raise ValueError(
            "Arena geometry must use the recording-level video and sidecars "
            "from the recording's cams directory."
        )
    expected_summary = camera_dir / f"{source.video_path.stem}_external_summary.json"
    expected_keyframes = camera_dir / f"{source.video_path.stem}_keyframe.json"
    if source.summary_path != expected_summary:
        raise ValueError("Arena-geometry summary must match the source-video basename.")
    if source.keyframe_path != expected_keyframes:
        raise ValueError(
            "Arena-geometry keyframe summary must match the source-video basename."
        )


@dataclass(frozen=True)
class ArenaGeometryReviewFragmentInputs:
    workflow_id: str
    target_id: str
    recording_dir: Path
    analysis_zarr: Path
    recovery_receipt_path: Path | None
    source: ArenaGeometryProbeSource
    repo: Path
    run_root: Path
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    max_keyframes_per_window: int = 21
    span_seconds: float = 5.0
    coarse_max_dimension_px: int = 2048
    geometry_source: str = "recovery-receipt"
    geometry_camera_serial: str | None = None
    geometry_arena_id: str | None = None
    citrus_h5_path: Path | None = None
    acquisition_resources: LsfResources = LsfResources(
        queue="short", ncores=2, mem_gb=8, walltime="1:00", span_hosts=1
    )
    probe_resources: LsfResources = LsfResources(
        queue="gpu_l4",
        ncores=8,
        mem_gb=32,
        gpus=1,
        walltime="1:00",
        span_hosts=1,
    )

    def __post_init__(self) -> None:
        target_id = str(self.target_id).strip()
        workflow_id = str(self.workflow_id).strip()
        if not target_id or not workflow_id:
            raise ValueError("Arena-geometry target and workflow IDs cannot be empty.")
        recording = self.recording_dir.expanduser().resolve()
        analysis = self.analysis_zarr.expanduser().resolve()
        repo = self.repo.expanduser().resolve()
        run_root = self.run_root.expanduser().resolve()
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
                raise ValueError(
                    "Historical recovery geometry requires a recovery receipt."
                )
            receipt = _require_file(
                self.recovery_receipt_path,
                label="recording geometry recovery receipt",
            )
        elif self.recovery_receipt_path is not None:
            raise ValueError(
                "Producer-native geometry must not be configured with a recovery receipt."
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
                "Producer-native geometry requires an exact camera serial and arena ID."
            )
        citrus_h5 = None
        if geometry_source == "citrus-h5":
            if self.citrus_h5_path is None:
                raise ValueError("citrus-h5 geometry requires an exact Citrus H5 path.")
            citrus_h5 = _require_file(
                self.citrus_h5_path,
                label="recording-bound Citrus H5",
            )
        elif self.citrus_h5_path is not None:
            raise ValueError(
                "A Citrus H5 path is only valid for the citrus-h5 geometry source."
            )
        if not recording.is_dir() or not analysis.is_dir():
            raise FileNotFoundError("Recording directory and analysis Zarr must exist.")
        if not (analysis / "zarr.json").is_file():
            raise FileNotFoundError(f"Analysis target is not Zarr v3: {analysis}")
        recording_bound_paths = {"analysis Zarr": analysis}
        if receipt is not None:
            recording_bound_paths["recovery receipt"] = receipt
        if citrus_h5 is not None:
            recording_bound_paths["Citrus H5"] = citrus_h5
        for label, path in recording_bound_paths.items():
            try:
                path.relative_to(recording)
            except ValueError as exc:
                raise ValueError(
                    f"Arena-geometry {label} must belong to the target recording."
                ) from exc
        validate_recording_level_probe_source(recording, self.source)
        if not (repo / "scripts" / "py").is_file():
            raise FileNotFoundError(f"Palette repository is invalid: {repo}")
        if int(self.max_keyframes_per_window) < 3:
            raise ValueError("max_keyframes_per_window must be at least three.")
        if float(self.span_seconds) <= 0 or int(self.coarse_max_dimension_px) < 256:
            raise ValueError("Probe span and coarse dimension must be positive.")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "workflow_id", workflow_id)
        object.__setattr__(self, "recording_dir", recording)
        object.__setattr__(self, "analysis_zarr", analysis)
        object.__setattr__(self, "recovery_receipt_path", receipt)
        object.__setattr__(self, "geometry_source", geometry_source)
        object.__setattr__(self, "geometry_camera_serial", camera_serial or None)
        object.__setattr__(self, "geometry_arena_id", arena_id or None)
        object.__setattr__(self, "citrus_h5_path", citrus_h5)
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "run_root", run_root)


@dataclass(frozen=True)
class ArenaGeometryReviewFragmentOutputs:
    target_id: str
    acquisition_candidate_id: str
    acquisition_candidate_path: Path
    acquisition_receipt_path: Path
    review_package_dir: Path
    fit_report_path: Path
    review_montage_path: Path
    review_receipt_path: Path
    terminal_job_keys: tuple[str, ...]
    acquisition_artifact_key: str
    review_artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "acquisition_candidate_id": self.acquisition_candidate_id,
            "acquisition_candidate_path": str(self.acquisition_candidate_path),
            "acquisition_receipt_path": str(self.acquisition_receipt_path),
            "review_package_dir": str(self.review_package_dir),
            "fit_report_path": str(self.fit_report_path),
            "review_montage_path": str(self.review_montage_path),
            "review_receipt_path": str(self.review_receipt_path),
            "terminal_job_keys": list(self.terminal_job_keys),
            "acquisition_artifact_key": self.acquisition_artifact_key,
            "review_artifact_key": self.review_artifact_key,
            "human_review_status": "required",
            "operational_selection": "not_performed",
            "detection_gate_applied": False,
            "registry_update": False,
        }


@dataclass(frozen=True)
class ArenaGeometryReviewWorkflowModule:
    fragment: LsfWorkflowFragment
    outputs: ArenaGeometryReviewFragmentOutputs


@dataclass(frozen=True)
class ArenaGeometryReviewArrayWorkflowModule:
    """Campaign-level LSF arrays for independent pre-review target tasks."""

    fragment: LsfWorkflowFragment
    outputs: tuple[ArenaGeometryReviewFragmentOutputs, ...]


@dataclass(frozen=True)
class _ArenaGeometryReviewTargetPlan:
    target_safe: str
    acquisition_candidate_id: str
    acquisition_candidate_path: Path
    acquisition_result_path: Path
    acquisition_scratch: str
    acquisition_command: tuple[str, ...]
    review_dir: Path
    probe_command: tuple[str, ...]

    def outputs(
        self,
        *,
        target_id: str,
        acquisition_job_key: str,
        probe_job_key: str,
    ) -> ArenaGeometryReviewFragmentOutputs:
        acquisition_artifact = (
            f"arena_geometry_acquisition_candidate:{self.target_safe}"
        )
        review_artifact = f"arena_geometry_review_package:{self.target_safe}"
        return ArenaGeometryReviewFragmentOutputs(
            target_id=target_id,
            acquisition_candidate_id=self.acquisition_candidate_id,
            acquisition_candidate_path=self.acquisition_candidate_path,
            acquisition_receipt_path=self.acquisition_result_path,
            review_package_dir=self.review_dir,
            fit_report_path=self.review_dir / "fit_report.json",
            review_montage_path=self.review_dir / "dish_rim_review_montage.png",
            review_receipt_path=self.review_dir / "review_package.json",
            terminal_job_keys=(acquisition_job_key, probe_job_key),
            acquisition_artifact_key=acquisition_artifact,
            review_artifact_key=review_artifact,
        )


def _plan_review_target(
    inputs: ArenaGeometryReviewFragmentInputs,
    *,
    array_indexed: bool,
) -> _ArenaGeometryReviewTargetPlan:
    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    if inputs.geometry_source == "recovery-receipt":
        assert inputs.recovery_receipt_path is not None
        candidate = plan_recovered_acquisition_geometry_candidate(
            source_zarr=inputs.analysis_zarr,
            receipt_path=inputs.recovery_receipt_path,
        )
    else:
        assert inputs.geometry_camera_serial is not None
        assert inputs.geometry_arena_id is not None
        candidate = plan_producer_native_acquisition_geometry_candidate(
            source_zarr=inputs.analysis_zarr,
            camera_serial=inputs.geometry_camera_serial,
            arena_id=inputs.geometry_arena_id,
            recording_folder=(
                inputs.recording_dir
                if inputs.geometry_source == "producer-folder"
                else None
            ),
            citrus_h5=inputs.citrus_h5_path,
        )
    acquisition_result = (
        inputs.run_root / "arena_geometry" / target_safe / "acquisition_candidate.json"
    )
    work_unit = (
        f"{RUNTIME_JOB_ID_TOKEN}_{RUNTIME_JOB_INDEX_TOKEN}"
        if array_indexed
        else RUNTIME_JOB_ID_TOKEN
    )
    acquisition_scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{work_unit}/arena_geometry_acquisition"
    )
    publish_command = [
        "scripts/py",
        "-m",
        "fisheye.utils.publish_acquisition_geometry_candidates",
        "--recording",
        str(inputs.recording_dir),
        "--analysis-zarr",
        str(inputs.analysis_zarr),
        "--geometry-source",
        inputs.geometry_source,
        "--scratch-root",
        acquisition_scratch,
        "--result-json",
        str(acquisition_result),
        "--apply",
    ]
    if inputs.geometry_source != "recovery-receipt":
        assert inputs.geometry_camera_serial is not None
        assert inputs.geometry_arena_id is not None
        publish_command.extend(
            (
                "--camera-serial",
                inputs.geometry_camera_serial,
                "--arena-id",
                inputs.geometry_arena_id,
            )
        )
    if inputs.citrus_h5_path is not None:
        publish_command.extend(("--citrus-h5", str(inputs.citrus_h5_path)))
    acquisition_command = chain_commands(
        (("mkdir", "-p", acquisition_scratch), tuple(publish_command))
    )
    review_dir = inputs.run_root / "arena_geometry" / target_safe / "review_package"
    probe_command = [
        "scripts/py",
        "-m",
        "fisheye.diagnostics.probe_recording_dish_rim_fit",
        "--video",
        str(inputs.source.video_path),
        "--summary",
        str(inputs.source.summary_path),
        "--keyframes",
        str(inputs.source.keyframe_path),
        "--output-dir",
        str(review_dir),
        "--max-keyframes-per-window",
        str(int(inputs.max_keyframes_per_window)),
        "--span-seconds",
        str(float(inputs.span_seconds)),
        "--coarse-max-dimension-px",
        str(int(inputs.coarse_max_dimension_px)),
    ]
    if inputs.source.acquisition_observation_path is not None:
        probe_command.extend(
            (
                "--acquisition-observation",
                str(inputs.source.acquisition_observation_path),
            )
        )
    return _ArenaGeometryReviewTargetPlan(
        target_safe=target_safe,
        acquisition_candidate_id=candidate.candidate_id,
        acquisition_candidate_path=candidate.target_run_path,
        acquisition_result_path=acquisition_result,
        acquisition_scratch=acquisition_scratch,
        acquisition_command=acquisition_command,
        review_dir=review_dir,
        probe_command=tuple(probe_command),
    )


def build_arena_geometry_review_fragment(
    inputs: ArenaGeometryReviewFragmentInputs,
) -> ArenaGeometryReviewWorkflowModule:
    """Build acquisition publication and blind fit jobs that stop for review."""

    planned = _plan_review_target(inputs, array_indexed=False)
    acquisition_key = f"arena_geometry_acquisition:{planned.target_safe}"
    acquisition_job = build_job(
        workflow_id=inputs.workflow_id,
        family=FAMILY,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=acquisition_key,
        stage="arena_geometry_acquisition_candidate",
        command=planned.acquisition_command,
        resources=inputs.acquisition_resources,
        upstream=inputs.upstream_job_keys,
        expected_outputs=(
            planned.acquisition_candidate_path / "zarr.json",
            planned.acquisition_result_path,
        ),
        cleanup_paths=(planned.acquisition_scratch,),
    )

    probe_key = f"arena_geometry_probe:{planned.target_safe}"
    probe_job = build_job(
        workflow_id=inputs.workflow_id,
        family=FAMILY,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=probe_key,
        stage="arena_geometry_blind_keyframe_probe",
        command=planned.probe_command,
        resources=inputs.probe_resources,
        upstream=inputs.upstream_job_keys,
        expected_outputs=(
            planned.review_dir / "fit_report.json",
            planned.review_dir / "dish_rim_review_montage.png",
            planned.review_dir / "review_package.json",
        ),
    )

    outputs = planned.outputs(
        target_id=inputs.target_id,
        acquisition_job_key=acquisition_key,
        probe_job_key=probe_key,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"arena_geometry_review:{planned.target_safe}",
        jobs=(acquisition_job, probe_job),
        requires=inputs.required_artifacts,
        provides=(outputs.acquisition_artifact_key, outputs.review_artifact_key),
        metadata={
            "module": "arena_geometry_review",
            "target_id": inputs.target_id,
            "geometry_scope": "recording_level",
            "downstream_layouts": ["clipped", "whole_recording"],
            "human_review_barrier": True,
            "selection_activation": "deferred",
            "registry_update": False,
            "geometry_source": inputs.geometry_source,
            "source": inputs.source.to_json(),
            "outputs": outputs.to_json(),
        },
    )
    return ArenaGeometryReviewWorkflowModule(fragment=fragment, outputs=outputs)


def build_arena_geometry_review_array_fragment(
    inputs: tuple[ArenaGeometryReviewFragmentInputs, ...],
    *,
    acquisition_max_concurrent: int = 8,
    probe_max_concurrent: int = 4,
) -> ArenaGeometryReviewArrayWorkflowModule:
    """Pack one or many pre-review targets into two independent LSF arrays."""

    if not inputs:
        raise ValueError("Arena-geometry array fragment requires at least one target.")
    first = inputs[0]
    shared_fields = (
        "workflow_id",
        "repo",
        "run_root",
        "upstream_job_keys",
        "acquisition_resources",
        "probe_resources",
    )
    for item in inputs[1:]:
        for field in shared_fields:
            if getattr(item, field) != getattr(first, field):
                raise ValueError(
                    f"Arena-geometry array targets require one shared {field}."
                )
    if int(acquisition_max_concurrent) <= 0 or int(probe_max_concurrent) <= 0:
        raise ValueError("Arena-geometry array concurrency must be positive.")

    acquisition_tasks = []
    probe_tasks = []
    outputs: list[ArenaGeometryReviewFragmentOutputs] = []
    required_artifacts: list[str] = []
    provided_artifacts: list[str] = []
    for item in inputs:
        planned = _plan_review_target(item, array_indexed=True)
        acquisition_task_key = f"arena_geometry_acquisition:{planned.target_safe}"
        acquisition_tasks.append(
            build_execution_task(
                run_root=item.run_root,
                task_key=acquisition_task_key,
                stage="arena_geometry_acquisition_candidate",
                command=planned.acquisition_command,
                expected_outputs=(
                    planned.acquisition_candidate_path / "zarr.json",
                    planned.acquisition_result_path,
                ),
                cleanup_paths=(planned.acquisition_scratch,),
                array_indexed=True,
            )
        )

        probe_task_key = f"arena_geometry_probe:{planned.target_safe}"
        probe_tasks.append(
            build_execution_task(
                run_root=item.run_root,
                task_key=probe_task_key,
                stage="arena_geometry_blind_keyframe_probe",
                command=planned.probe_command,
                expected_outputs=(
                    planned.review_dir / "fit_report.json",
                    planned.review_dir / "dish_rim_review_montage.png",
                    planned.review_dir / "review_package.json",
                ),
                array_indexed=True,
            )
        )

        output = planned.outputs(
            target_id=item.target_id,
            acquisition_job_key="arena_geometry_acquisition_array",
            probe_job_key="arena_geometry_probe_array",
        )
        outputs.append(output)
        required_artifacts.extend(item.required_artifacts)
        provided_artifacts.extend(
            (output.acquisition_artifact_key, output.review_artifact_key)
        )

    acquisition_job = build_task_group_job(
        workflow_id=first.workflow_id,
        family=FAMILY,
        repo=first.repo,
        run_root=first.run_root,
        job_key="arena_geometry_acquisition_array",
        stage="arena_geometry_acquisition_candidate",
        tasks=tuple(acquisition_tasks),
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=min(int(acquisition_max_concurrent), len(acquisition_tasks)),
        resources=first.acquisition_resources,
        upstream=first.upstream_job_keys,
    )
    probe_job = build_task_group_job(
        workflow_id=first.workflow_id,
        family=FAMILY,
        repo=first.repo,
        run_root=first.run_root,
        job_key="arena_geometry_probe_array",
        stage="arena_geometry_blind_keyframe_probe",
        tasks=tuple(probe_tasks),
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=min(int(probe_max_concurrent), len(probe_tasks)),
        resources=first.probe_resources,
        upstream=first.upstream_job_keys,
    )
    fragment = LsfWorkflowFragment(
        fragment_id="arena_geometry_review_array",
        jobs=(acquisition_job, probe_job),
        requires=tuple(dict.fromkeys(required_artifacts)),
        provides=tuple(provided_artifacts),
        metadata={
            "module": "arena_geometry_review_array",
            "target_count": len(inputs),
            "geometry_scope": "recording_level",
            "downstream_layouts": ["clipped", "whole_recording"],
            "execution_mode": "lsf_arrays",
            "arrays_independent": True,
            "human_review_barrier": True,
            "selection_activation": "deferred",
            "registry_update": False,
            "outputs": [item.to_json() for item in outputs],
        },
    )
    return ArenaGeometryReviewArrayWorkflowModule(
        fragment=fragment,
        outputs=tuple(outputs),
    )


def _validate_review_package(
    *, review_receipt_path: Path, fit_report_path: Path, montage_path: Path
) -> Mapping[str, Any]:
    receipt = json.loads(review_receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_id") != REVIEW_PACKAGE_SCHEMA_ID
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "awaiting_explicit_human_review"
    ):
        raise ValueError("Arena-geometry review package is invalid.")
    fit = receipt.get("fit_report")
    montage = receipt.get("montage")
    if not isinstance(fit, Mapping) or not isinstance(montage, Mapping):
        raise ValueError("Arena-geometry review package lacks bound artifacts.")
    if fit.get("sha256") != _sha256_file(fit_report_path):
        raise ValueError("Arena-geometry fit report changed after review packaging.")
    if montage.get("sha256") != _sha256_file(montage_path):
        raise ValueError("Arena-geometry review montage changed after packaging.")
    return receipt


@dataclass(frozen=True)
class ReviewedArenaGeometryCandidateFragmentInputs:
    workflow_id: str
    target_id: str
    analysis_zarr: Path
    fit_report_path: Path
    review_montage_path: Path
    review_receipt_path: Path
    reviewer: str
    reviewed_at_utc: str
    repo: Path
    run_root: Path
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    resources: LsfResources = LsfResources(
        queue="short", ncores=2, mem_gb=8, walltime="1:00", span_hosts=1
    )


@dataclass(frozen=True)
class ReviewedArenaGeometryCandidateWorkflowModule:
    fragment: LsfWorkflowFragment
    candidate_id: str
    candidate_path: Path
    receipt_path: Path
    artifact_key: str
    terminal_job_key: str


def build_reviewed_arena_geometry_candidate_fragment(
    inputs: ReviewedArenaGeometryCandidateFragmentInputs,
) -> ReviewedArenaGeometryCandidateWorkflowModule:
    """Build the separate post-review candidate publication node."""

    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    zarr_path = inputs.analysis_zarr.expanduser().resolve()
    report = _require_file(inputs.fit_report_path, label="fit report")
    montage = _require_file(inputs.review_montage_path, label="review montage")
    receipt = _require_file(inputs.review_receipt_path, label="review package receipt")
    _validate_review_package(
        review_receipt_path=receipt,
        fit_report_path=report,
        montage_path=montage,
    )
    plan = plan_reviewed_palette_geometry_candidate(
        source_zarr=zarr_path,
        fit_report_path=report,
        montage_path=montage,
        reviewer=inputs.reviewer,
        reviewed_at_utc=inputs.reviewed_at_utc,
    )
    result = inputs.run_root / "arena_geometry" / target_safe / "palette_candidate.json"
    scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "arena_geometry_palette_candidate"
    )
    job_key = f"arena_geometry_palette_candidate:{target_safe}"
    command = chain_commands(
        (
            ("mkdir", "-p", scratch),
            (
                "scripts/py",
                "-m",
                "fisheye.utils.publish_reviewed_palette_geometry_candidate",
                "--zarr",
                str(zarr_path),
                "--fit-report",
                str(report),
                "--review-montage",
                str(montage),
                "--reviewer",
                str(inputs.reviewer),
                "--reviewed-at-utc",
                str(inputs.reviewed_at_utc),
                "--scratch-root",
                scratch,
                "--result-json",
                str(result),
                "--apply",
            ),
        )
    )
    job = build_job(
        workflow_id=inputs.workflow_id,
        family=FAMILY,
        repo=inputs.repo.expanduser().resolve(),
        run_root=inputs.run_root.expanduser().resolve(),
        job_key=job_key,
        stage="arena_geometry_reviewed_palette_candidate",
        command=command,
        resources=inputs.resources,
        upstream=inputs.upstream_job_keys,
        expected_outputs=(plan.target_run_path / "zarr.json", result),
        cleanup_paths=(scratch,),
    )
    artifact_key = f"arena_geometry_palette_candidate:{target_safe}"
    fragment = LsfWorkflowFragment(
        fragment_id=f"arena_geometry_palette_candidate:{target_safe}",
        jobs=(job,),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "reviewed_arena_geometry_candidate",
            "target_id": inputs.target_id,
            "candidate_id": plan.candidate_id,
            "review_receipt_path": str(receipt),
            "selection_activation": "deferred",
            "detection_gate_applied": False,
            "registry_update": False,
        },
    )
    return ReviewedArenaGeometryCandidateWorkflowModule(
        fragment=fragment,
        candidate_id=plan.candidate_id,
        candidate_path=plan.target_run_path,
        receipt_path=result,
        artifact_key=artifact_key,
        terminal_job_key=job_key,
    )


def compose_arena_geometry_workflow(
    *,
    workflow_id: str,
    modules: tuple[
        ArenaGeometryReviewWorkflowModule
        | ArenaGeometryReviewArrayWorkflowModule
        | ReviewedArenaGeometryCandidateWorkflowModule,
        ...,
    ],
    external_inputs: tuple[str, ...] = (),
) -> LsfWorkflow:
    if not modules:
        raise ValueError("An arena-geometry workflow requires at least one module.")
    target_count = sum(
        (
            len(module.outputs)
            if isinstance(module, ArenaGeometryReviewArrayWorkflowModule)
            else 1
        )
        for module in modules
    )
    return compose_lsf_workflow(
        workflow_id=workflow_id,
        family=FAMILY,
        fragments=tuple(module.fragment for module in modules),
        external_inputs=external_inputs,
        metadata={
            "workflow_scope": "recording_level_arena_geometry",
            "target_count": target_count,
            "human_review_barrier": True,
            "selection_activation": "deferred",
            "registry_update": False,
        },
    )


__all__ = [
    "ArenaGeometryProbeSource",
    "ArenaGeometryReviewFragmentInputs",
    "ArenaGeometryReviewFragmentOutputs",
    "ArenaGeometryReviewWorkflowModule",
    "ArenaGeometryReviewArrayWorkflowModule",
    "ReviewedArenaGeometryCandidateFragmentInputs",
    "ReviewedArenaGeometryCandidateWorkflowModule",
    "build_arena_geometry_review_fragment",
    "build_arena_geometry_review_array_fragment",
    "build_reviewed_arena_geometry_candidate_fragment",
    "compose_arena_geometry_workflow",
    "validate_recording_level_probe_source",
]
