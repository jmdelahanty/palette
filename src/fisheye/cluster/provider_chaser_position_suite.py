"""Exact LSF DAG for one selector-ineligible chaser position suite."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Sequence

from fisheye.analysis.chaser_profiles import (
    load_chaser_analysis_profile,
    resolve_chaser_analysis_modules,
    validate_chaser_runner_modules,
)
from fisheye.cluster.clipped_lsf import build_job
from fisheye.cluster.lsf import (
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN, RUNTIME_USER_TOKEN


FAMILY = "provider_chaser_position_suite"
PROFILE_ID = "chaser_position_suite_v1"
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _name(value: object, *, field: str) -> str:
    if type(value) is not str or _NAME_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be one exact bare non-selector name.")
    if value in {"latest", "latest_complete", "selected", "current"}:
        raise ValueError(f"{field} must be one exact bare non-selector name.")
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _commit(value: object) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError("palette_commit must be one full lowercase Git SHA.")
    return text


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive.")
    return result


def _epoch_roles(
    values: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    roles = tuple(values)
    if not roles:
        raise ValueError("At least one explicit epoch-role binding is required.")
    names = []
    windows = []
    for role, window in roles:
        role_name = _name(role, field="epoch role")
        if type(window) is not int or window < 0:
            raise ValueError("epoch window IDs must be non-negative integers.")
        names.append(role_name)
        windows.append(window)
    if len(set(names)) != len(names) or len(set(windows)) != len(windows):
        raise ValueError("Epoch roles and window IDs must each be unique.")
    return tuple(zip(names, windows, strict=True))


@dataclass(frozen=True, slots=True)
class ProviderChaserPositionSuiteWorkflowPlan:
    workflow: LsfWorkflow
    palette_commit: str
    analysis_zarr: Path
    run_name: str
    profile_path: Path
    profile_sha256: str
    publication_result_path: Path
    readiness_receipt_path: Path
    run_root: Path

    def to_json(self) -> dict[str, object]:
        return {
            "schema_id": "palette.provider_chaser_position_suite_workflow_plan",
            "schema_version": 1,
            "palette_commit": self.palette_commit,
            "analysis_zarr": str(self.analysis_zarr),
            "run_name": self.run_name,
            "profile_id": PROFILE_ID,
            "profile_path": str(self.profile_path),
            "profile_sha256": self.profile_sha256,
            "publication_result_path": str(self.publication_result_path),
            "readiness_receipt_path": str(self.readiness_receipt_path),
            "run_root": str(self.run_root),
            "selector_eligible": False,
            "production_selector_activation": False,
            "registry_update": False,
            "workflow": self.workflow.to_json(),
        }


def build_provider_chaser_position_suite_workflow(
    *,
    workflow_id: str,
    repo: str | Path,
    run_root: str | Path,
    analysis_zarr: str | Path,
    run_name: str,
    provider_run: str,
    geometry_selection_run: str,
    expected_selection_record_sha256: str,
    expected_physical_authority_sha256: str,
    epoch_role_bindings: Sequence[tuple[str, int]],
    analysis_profile_path: str | Path,
    palette_commit: str,
    expected_recording_id: str | None = None,
    treatment_role: str = "aggressive",
    baseline_role: str = "inert",
    radial_bin_width_mm: float = 2.0,
    near_zone_radius_mm: float = 5.0,
    near_entry_radius_mm: float = 5.0,
    near_exit_radius_mm: float = 6.0,
    perimeter_band_mm: float = 5.0,
    min_expected_count: float = 5.0,
    resources: LsfResources | None = None,
) -> ProviderChaserPositionSuiteWorkflowPlan:
    """Build publication -> bounded-readiness without selector or registry jobs."""

    workflow_id = _name(workflow_id, field="workflow_id")
    run_name = _name(run_name, field="run_name")
    provider_run = _name(provider_run, field="provider_run")
    geometry_selection_run = _name(
        geometry_selection_run, field="geometry_selection_run"
    )
    treatment_role = _name(treatment_role, field="treatment_role")
    baseline_role = _name(baseline_role, field="baseline_role")
    if treatment_role == baseline_role:
        raise ValueError("Treatment and baseline roles must differ.")
    roles = _epoch_roles(epoch_role_bindings)
    selection_sha256 = _digest(
        expected_selection_record_sha256,
        field="expected_selection_record_sha256",
    )
    physical_sha256 = _digest(
        expected_physical_authority_sha256,
        field="expected_physical_authority_sha256",
    )
    radial_bin_width_mm = _finite_positive(
        radial_bin_width_mm, field="radial_bin_width_mm"
    )
    near_zone_radius_mm = _finite_positive(
        near_zone_radius_mm, field="near_zone_radius_mm"
    )
    near_entry_radius_mm = _finite_positive(
        near_entry_radius_mm, field="near_entry_radius_mm"
    )
    near_exit_radius_mm = _finite_positive(
        near_exit_radius_mm, field="near_exit_radius_mm"
    )
    perimeter_band_mm = _finite_positive(perimeter_band_mm, field="perimeter_band_mm")
    min_expected_count = _finite_positive(
        min_expected_count, field="min_expected_count"
    )
    if near_exit_radius_mm <= near_entry_radius_mm:
        raise ValueError("near_exit_radius_mm must exceed near_entry_radius_mm.")
    if expected_recording_id is not None:
        expected_recording_id = _name(
            expected_recording_id, field="expected_recording_id"
        )
    commit = _commit(palette_commit)
    repo_path = Path(repo).expanduser().resolve()
    archive = Path(analysis_zarr).expanduser().resolve()
    root = Path(run_root).expanduser().resolve()
    profile_path = Path(analysis_profile_path).expanduser().resolve()
    if not repo_path.is_dir() or not (repo_path / "scripts/py").is_file():
        raise FileNotFoundError(f"Palette deployment is invalid: {repo_path}")
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr is unavailable: {archive}")
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Chaser analysis profile is unavailable: {profile_path}"
        )
    if root == archive or archive in root.parents:
        raise ValueError("Workflow run_root must remain outside the analysis Zarr.")
    profile = load_chaser_analysis_profile(profile_path)
    if profile.profile_id != PROFILE_ID or profile.profile_scope != "reduced":
        raise ValueError(f"Workflow requires exact reduced profile {PROFILE_ID!r}.")
    selected_modules = resolve_chaser_analysis_modules(profile)
    validate_chaser_runner_modules(selected_modules)
    if [module.module_id for module in selected_modules] != [
        "stimulus_epochs",
        "provider_chaser_position_suite",
    ]:
        raise ValueError("Reduced position-suite profile module closure is invalid.")
    if (archive / "analysis/provider_chaser_position_suite_runs" / run_name).exists():
        raise FileExistsError("Refusing to plan over an existing immutable suite run.")

    resources = resources or LsfResources(
        queue="short",
        ncores=1,
        mem_gb=8,
        gpus=0,
        walltime="1:00",
        span_hosts=1,
    )
    publication_result = root / "publication_result.json"
    readiness_receipt = root / "readiness_receipt.json"
    scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "provider_chaser_position_suite"
    )
    publication_key = f"provider_chaser_position_suite:{run_name}"
    readiness_key = f"provider_chaser_position_suite_readiness:{run_name}"
    command = [
        "scripts/py",
        "-m",
        "fisheye.utils.materialize_provider_chaser_position_suite",
        str(archive),
        "--run-name",
        run_name,
        "--provider-run",
        provider_run,
        "--geometry-selection-run",
        geometry_selection_run,
        "--expected-selection-record-sha256",
        selection_sha256,
        "--expected-physical-authority-sha256",
        physical_sha256,
        "--treatment-role",
        treatment_role,
        "--baseline-role",
        baseline_role,
        "--radial-bin-width-mm",
        format(radial_bin_width_mm, ".17g"),
        "--near-zone-radius-mm",
        format(near_zone_radius_mm, ".17g"),
        "--near-entry-radius-mm",
        format(near_entry_radius_mm, ".17g"),
        "--near-exit-radius-mm",
        format(near_exit_radius_mm, ".17g"),
        "--perimeter-band-mm",
        format(perimeter_band_mm, ".17g"),
        "--min-expected-count",
        format(min_expected_count, ".17g"),
        "--scratch-root",
        scratch,
        "--output-json",
        str(publication_result),
        "--apply",
    ]
    if expected_recording_id is not None:
        command.extend(("--expected-recording-id", expected_recording_id))
    for role, window in roles:
        command.extend(("--epoch-role", f"{role}={window}"))
    publication_job = build_job(
        workflow_id=workflow_id,
        family=FAMILY,
        repo=repo_path,
        run_root=root,
        job_key=publication_key,
        stage="provider_chaser_position_suite_candidate",
        command=command,
        resources=resources,
        expected_outputs=(
            archive
            / "analysis/provider_chaser_position_suite_runs"
            / run_name
            / "zarr.json",
            publication_result,
        ),
        cleanup_paths=(scratch,),
    )
    readiness_command = [
        "scripts/py",
        "-m",
        "fisheye.utils.provider_chaser_position_suite_readiness",
        str(archive),
        "--run-name",
        run_name,
        "--output-json",
        str(readiness_receipt),
    ]
    if expected_recording_id is not None:
        readiness_command.extend(("--expected-recording-id", expected_recording_id))
    readiness_job = build_job(
        workflow_id=workflow_id,
        family=FAMILY,
        repo=repo_path,
        run_root=root,
        job_key=readiness_key,
        stage="provider_chaser_position_suite_readiness",
        command=readiness_command,
        resources=resources,
        upstream=(publication_key,),
        expected_outputs=(readiness_receipt,),
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"provider_chaser_position_suite:{run_name}",
        jobs=(publication_job, readiness_job),
        requires=(f"provider_chaser_distance:{provider_run}",),
        provides=(
            f"provider_chaser_position_suite_candidate:{run_name}",
            f"provider_chaser_position_suite_readiness:{run_name}",
        ),
        metadata={
            "profile_id": profile.profile_id,
            "profile_sha256": profile.sha256,
            "selector_eligible": False,
            "production_selector_activation": False,
            "registry_update": False,
        },
    )
    workflow = compose_lsf_workflow(
        workflow_id=workflow_id,
        family=FAMILY,
        fragments=(fragment,),
        external_inputs=(f"provider_chaser_distance:{provider_run}",),
        metadata={
            "analysis_zarr": str(archive),
            "palette_repo": str(repo_path),
            "palette_commit": commit,
            "profile_id": profile.profile_id,
            "profile_sha256": profile.sha256,
            "workflow_scope": "reduced_provider_chaser_position_only",
            "selector_eligible": False,
            "production_selector_activation": False,
            "registry_update": False,
            "required_ci_before_promotion": True,
        },
    )
    return ProviderChaserPositionSuiteWorkflowPlan(
        workflow=workflow,
        palette_commit=commit,
        analysis_zarr=archive,
        run_name=run_name,
        profile_path=profile_path,
        profile_sha256=profile.sha256,
        publication_result_path=publication_result,
        readiness_receipt_path=readiness_receipt,
        run_root=root,
    )


__all__ = [
    "FAMILY",
    "PROFILE_ID",
    "ProviderChaserPositionSuiteWorkflowPlan",
    "build_provider_chaser_position_suite_workflow",
]
