"""One typed admission resolver for validated-behavior source receipts.

Membership planning and later current-source validation both call this module.
Each installed receipt profile owns one full-strength branch; callers never
guess a profile from path names and never fall back after a selected branch
fails.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from fisheye.analysis_workflows.exact_chaser_projection_receipt import (
    validate_exact_chaser_projection_receipt,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .contracts import TemporalPolicy

EXACT_CHASER_ADMISSION_ROLE = "exact_chaser_projection"
CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE = "core_behavior_workflow_execution"

CORE_BEHAVIOR_EXECUTION_SCHEMA_ID = "palette.analysis_workflow_execution"
CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION = 3
CORE_BEHAVIOR_WORKFLOW_ID = "core_behavior_v1"

CORE_BEHAVIOR_REQUIRED_STAGE_NODES = (
    "track_kinematics",
    "swim_bouts",
    "subject_shape",
    "eye_angles",
    "tail_kinematics",
)

_REPORT_FIELDS = {
    "schema_id",
    "schema_version",
    "execution_id",
    "mode",
    "status",
    "created_at_utc",
    "completed_at_utc",
    "host",
    "lsf_job_id",
    "palette_git",
    "zarr_path",
    "registry_write_mode",
    "workflow",
    "execution_plan",
    "node_results",
    "error",
}
_GIT_FIELDS = {
    "branch",
    "commit_hash",
    "dirty_files",
    "is_dirty",
    "remote_url",
    "short_hash",
    "top_level",
}
_WORKFLOW_FIELDS = {
    "description",
    "nodes",
    "run_selection",
    "schema_id",
    "schema_version",
    "targets",
    "temporal_policy",
    "workflow_id",
}
_WORKFLOW_NODE_FIELDS = {
    "depends_on",
    "description",
    "execution_policy",
    "id",
    "kind",
    "output_run_from",
    "runnable",
    "stage_id",
    "temporal_product",
}
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_NODE_PATH_PREFIXES = {
    "track_kinematics": "analysis/track_kinematics_runs/offline/",
    "swim_bouts": "analysis/swim_bout_runs/",
    "subject_shape": "analysis/subject_shape_runs/",
    "eye_angles": "analysis/eye_angle_runs/",
    "tail_kinematics": "analysis/tail_kinematics_runs/",
}


class ValidatedBehaviorAdmissionError(ValueError):
    """A selected source receipt cannot satisfy its declared profile."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorAdmissionError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _digest(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return text


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Admission receipt does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorAdmissionError(
            f"Cannot read admission receipt {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"Admission receipt must contain one JSON object: {path}")
    return value


def _exact_path(value: object, *, field: str) -> Path:
    raw = str(value) if isinstance(value, Path) else _text(value, field=field)
    path = Path(raw)
    resolved = path.expanduser().resolve(strict=False)
    if not path.is_absolute() or str(resolved) != raw:
        _fail(f"{field} must be one canonical absolute path.")
    return resolved


def _expected_binding(
    binding: Mapping[str, Any],
    *,
    path: Path,
    file_sha256: str,
    record_sha256: str,
    schema_id: str,
    schema_version: int,
) -> dict[str, Any]:
    expected = {
        "role": _text(binding.get("role"), field="admission receipt role"),
        "path": str(path),
        "file_sha256": file_sha256,
        "record_sha256": record_sha256,
        "schema_id": schema_id,
        "schema_version": schema_version,
    }
    if _plain(binding) != expected:
        _fail(f"Admission receipt binding is stale or inexact: {path}")
    return expected


def validate_core_behavior_execution_report(
    value: object,
    *,
    expected_analysis_zarr: str | Path,
    expected_recording_id: str,
) -> dict[str, Any]:
    """Validate one completed core-behavior execution report as admission.

    This is deliberately receipt-level validation. Scientific bundle building
    subsequently reopens every selected Zarr authority through its strict
    source resolver and seals those bindings into the bundle-set member.
    """

    report = _plain(_mapping(value, field="core-behavior execution report"))
    if set(report) != _REPORT_FIELDS:
        _fail("Core-behavior execution-report field set is inexact.")
    if (
        report.get("schema_id") != CORE_BEHAVIOR_EXECUTION_SCHEMA_ID
        or report.get("schema_version") != CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION
        or report.get("mode") != "apply"
        or report.get("status") != "complete"
        or report.get("error") is not None
        or report.get("registry_write_mode") != "deferred_to_serial_finalizer"
    ):
        _fail("Core-behavior execution report is not one completed deferred-write run.")

    expected_zarr = _exact_path(expected_analysis_zarr, field="expected analysis_zarr")
    observed_zarr = _exact_path(report.get("zarr_path"), field="report zarr_path")
    if observed_zarr != expected_zarr:
        _fail("Core-behavior execution report binds another analysis Zarr.")
    recording_id = _text(expected_recording_id, field="expected recording_id")
    derived_recording = observed_zarr.name.removesuffix(".zarr").removesuffix(
        "_analysis"
    )
    if derived_recording != recording_id:
        _fail("Core-behavior execution report path and recording identity disagree.")

    git = _mapping(report.get("palette_git"), field="palette_git")
    if (
        set(git) != _GIT_FIELDS
        or git.get("is_dirty") is not False
        or git.get("dirty_files") != []
        or type(git.get("commit_hash")) is not str
        or _COMMIT_RE.fullmatch(str(git.get("commit_hash"))) is None
        or git.get("short_hash") != str(git.get("commit_hash"))[:8]
    ):
        _fail(
            "Core-behavior execution report lacks clean commit-pinned software authority."
        )
    _exact_path(git.get("top_level"), field="palette_git.top_level")

    workflow_raw = _mapping(report.get("workflow"), field="workflow")
    if (
        set(workflow_raw) != _WORKFLOW_FIELDS
        or workflow_raw.get("schema_id") != "palette.analysis_workflow"
        or workflow_raw.get("schema_version") != 1
        or workflow_raw.get("workflow_id") != CORE_BEHAVIOR_WORKFLOW_ID
    ):
        _fail("Execution report is not one exact core-behavior workflow snapshot.")
    workflow_nodes_raw = workflow_raw.get("nodes")
    if not isinstance(workflow_nodes_raw, list):
        _fail("Core-behavior workflow snapshot lacks its node roster.")
    workflow_nodes: dict[str, Mapping[str, Any]] = {}
    for index, raw_node in enumerate(workflow_nodes_raw):
        node = _mapping(raw_node, field=f"workflow.nodes[{index}]")
        if set(node) != _WORKFLOW_NODE_FIELDS:
            _fail(f"Core-behavior workflow node {index} has an inexact field set.")
        node_id = _text(node.get("id"), field=f"workflow.nodes[{index}].id")
        if node_id in workflow_nodes or node.get("stage_id") not in {None, node_id}:
            _fail("Core-behavior workflow node identity is duplicated or inconsistent.")
        workflow_nodes[node_id] = node
    if not set(CORE_BEHAVIOR_REQUIRED_STAGE_NODES).issubset(workflow_nodes):
        _fail("Core-behavior workflow snapshot lacks required scientific stages.")
    run_selection = _mapping(workflow_raw.get("run_selection"), field="run_selection")
    for node_id in CORE_BEHAVIOR_REQUIRED_STAGE_NODES:
        _text(run_selection.get(node_id), field=f"run_selection.{node_id}")
    temporal_policy = _mapping(
        workflow_raw.get("temporal_policy"), field="workflow.temporal_policy"
    )
    if set(temporal_policy) != {
        "activity_spatial",
        "eye_traces",
        "kinematics",
        "tail_traces",
    }:
        _fail("Core-behavior workflow temporal-policy roster is inexact.")
    try:
        installed_temporal_policy = TemporalPolicy.from_mapping(temporal_policy)
    except ValueError as exc:
        _fail(f"Core-behavior workflow temporal policy is invalid: {exc}")
    if _plain(temporal_policy) != installed_temporal_policy.to_dict():
        _fail("Core-behavior workflow temporal policy is not canonical.")

    execution_plan = _mapping(report.get("execution_plan"), field="execution_plan")
    if (
        execution_plan.get("schema_id") != CORE_BEHAVIOR_EXECUTION_SCHEMA_ID
        or execution_plan.get("schema_version")
        != CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION
        or execution_plan.get("execution_id") != report.get("execution_id")
    ):
        _fail("Execution-plan identity differs from the report envelope.")
    workflow_plan = _mapping(
        execution_plan.get("workflow_plan"), field="execution_plan.workflow_plan"
    )
    if workflow_plan.get("workflow_id") != CORE_BEHAVIOR_WORKFLOW_ID or _plain(
        workflow_plan.get("temporal_policy")
    ) != _plain(temporal_policy):
        _fail("Execution plan binds another workflow.")

    raw_results = report.get("node_results")
    if not isinstance(raw_results, list):
        _fail("Core-behavior execution report lacks node results.")
    results: dict[str, Mapping[str, Any]] = {}
    for index, raw_result in enumerate(raw_results):
        result = _mapping(raw_result, field=f"node_results[{index}]")
        node_id = _text(result.get("node_id"), field=f"node_results[{index}].node_id")
        if node_id in results:
            _fail("Core-behavior execution report repeats a node result.")
        if result.get("stage_id") != node_id:
            _fail(f"Core-behavior node {node_id!r} has another stage identity.")
        if result.get("status") not in {"complete", "reused"}:
            _fail(f"Core-behavior node {node_id!r} did not complete successfully.")
        verification = _mapping(
            result.get("verification"), field=f"node_results[{index}].verification"
        )
        if (
            verification.get("available") is not True
            or verification.get("completion_status") != "complete"
            or verification.get("run_name") != result.get("run_name")
            or verification.get("stage_id") != result.get("stage_id")
        ):
            _fail(
                f"Core-behavior node {node_id!r} lacks exact successful verification."
            )
        results[node_id] = result

    missing = sorted(set(CORE_BEHAVIOR_REQUIRED_STAGE_NODES) - set(results))
    if missing:
        _fail(f"Core-behavior execution report lacks required nodes: {missing!r}.")

    plan_nodes_raw = workflow_plan.get("nodes")
    if not isinstance(plan_nodes_raw, list):
        _fail("Execution workflow plan lacks a node roster.")
    plan_nodes = {
        str(item.get("node_id")): item
        for item in plan_nodes_raw
        if isinstance(item, Mapping) and item.get("node_id")
    }
    output_runs = _mapping(execution_plan.get("output_runs"), field="output_runs")
    runs: dict[str, dict[str, Any]] = {}
    for node_id in CORE_BEHAVIOR_REQUIRED_STAGE_NODES:
        result = results[node_id]
        plan_node = _mapping(plan_nodes.get(node_id), field=f"plan node {node_id}")
        workflow_node = workflow_nodes[node_id]
        if plan_node.get("stage_id") != workflow_node.get("stage_id") or plan_node.get(
            "depends_on"
        ) != workflow_node.get("depends_on"):
            _fail(f"Core-behavior plan node {node_id!r} differs from its workflow.")
        run_name = _text(result.get("run_name"), field=f"{node_id} run_name")
        if "/" in run_name or "\\" in run_name:
            _fail(f"Core-behavior node {node_id!r} run name is not one child name.")
        expected_run = (
            plan_node.get("selected_run")
            if plan_node.get("action") == "reuse"
            else output_runs.get(node_id)
        )
        if expected_run != run_name:
            _fail(f"Core-behavior node {node_id!r} differs from its execution plan.")
        artifact_path = _text(
            _mapping(result.get("verification"), field="verification").get(
                "artifact_path"
            ),
            field=f"{node_id} artifact_path",
        )
        prefix = _NODE_PATH_PREFIXES[node_id]
        if artifact_path != f"{prefix}{run_name}":
            _fail(f"Core-behavior node {node_id!r} artifact path is inexact.")
        runs[node_id] = {
            "run_name": run_name,
            "run_path": artifact_path,
            "stage_id": result.get("stage_id"),
            "execution_status": result.get("status"),
            "verification_sha256": canonical_json_sha256(
                _plain(result["verification"])
            ),
        }

    return {
        "schema_id": CORE_BEHAVIOR_EXECUTION_SCHEMA_ID,
        "schema_version": CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION,
        "execution_id": _text(report.get("execution_id"), field="execution_id"),
        "recording_id": recording_id,
        "analysis_zarr": str(observed_zarr),
        "workflow_id": CORE_BEHAVIOR_WORKFLOW_ID,
        "workflow_sha256": canonical_json_sha256(_plain(workflow_raw)),
        "temporal_policy": _plain(temporal_policy),
        "execution_plan_sha256": canonical_json_sha256(_plain(execution_plan)),
        "palette_commit": str(git["commit_hash"]),
        "runs": runs,
        "record_sha256": canonical_json_sha256(report),
    }


def validate_admission_receipt_binding(
    binding: Mapping[str, Any],
    *,
    recording_id: str,
    analysis_zarr: str | Path,
) -> dict[str, Any]:
    """Resolve and fully validate one declared admission-receipt profile."""

    role = _text(binding.get("role"), field="admission receipt role")
    path = _exact_path(binding.get("path"), field="admission receipt path")
    raw = _read_json_object(path)
    observed_file_sha = sha256_file(path)
    if observed_file_sha != _digest(
        binding.get("file_sha256"), field="admission receipt file digest"
    ):
        _fail(f"Admission receipt file digest changed: {path}")

    if role == EXACT_CHASER_ADMISSION_ROLE:
        receipt = validate_exact_chaser_projection_receipt(
            raw,
            expected_analysis_zarr=str(analysis_zarr),
            expected_recording_id=recording_id,
            validate_current_metadata=False,
            validate_child_receipts=False,
        )
        return _expected_binding(
            binding,
            path=path,
            file_sha256=observed_file_sha,
            record_sha256=str(receipt["record_sha256"]),
            schema_id=str(receipt["schema_id"]),
            schema_version=int(receipt["schema_version"]),
        )
    if role == CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE:
        receipt = validate_core_behavior_execution_report(
            raw,
            expected_analysis_zarr=analysis_zarr,
            expected_recording_id=recording_id,
        )
        return _expected_binding(
            binding,
            path=path,
            file_sha256=observed_file_sha,
            record_sha256=str(receipt["record_sha256"]),
            schema_id=str(receipt["schema_id"]),
            schema_version=int(receipt["schema_version"]),
        )
    _fail(f"No installed admission resolver accepts role {role!r}.")


def bind_core_behavior_execution_report(
    path: str | Path,
    *,
    recording_id: str,
    analysis_zarr: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mint and validate a binding for one concrete execution report."""

    source = Path(path).expanduser().resolve()
    raw = _read_json_object(source)
    receipt = validate_core_behavior_execution_report(
        raw,
        expected_analysis_zarr=analysis_zarr,
        expected_recording_id=recording_id,
    )
    binding = {
        "role": CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
        "path": str(source),
        "file_sha256": sha256_file(source),
        "record_sha256": receipt["record_sha256"],
        "schema_id": receipt["schema_id"],
        "schema_version": receipt["schema_version"],
    }
    return (
        validate_admission_receipt_binding(
            binding,
            recording_id=recording_id,
            analysis_zarr=analysis_zarr,
        ),
        receipt,
    )


__all__ = [
    "CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE",
    "CORE_BEHAVIOR_EXECUTION_SCHEMA_ID",
    "CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION",
    "CORE_BEHAVIOR_REQUIRED_STAGE_NODES",
    "CORE_BEHAVIOR_WORKFLOW_ID",
    "EXACT_CHASER_ADMISSION_ROLE",
    "ValidatedBehaviorAdmissionError",
    "bind_core_behavior_execution_report",
    "sha256_file",
    "validate_admission_receipt_binding",
    "validate_core_behavior_execution_report",
]
