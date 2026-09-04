from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.analysis_workflows.core_behavior_cohort_adapter import (
    CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
    core_behavior_capability_contract,
)
from fisheye.analysis_workflows.validated_behavior_cohort_adapters import (
    RECORDING_BUNDLE_ADAPTER_ID,
)
from fisheye.analysis_workflows.validated_behavior_source_admission import (
    CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
    CORE_BEHAVIOR_REQUIRED_STAGE_NODES,
    EXACT_CHASER_ADMISSION_ROLE,
    ValidatedBehaviorAdmissionError,
    bind_core_behavior_execution_report,
    validate_core_behavior_execution_report,
)
from fisheye.analytics_exports.validated_behavior_cohort import (
    EXPORT_METHOD_ID,
    EXPORT_PLAN_METHOD_ID,
    validated_behavior_manifest_path,
)
import fisheye.analytics_exports.validated_behavior_cohort as export_core
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_TABLE_NAMES,
    validate_table_specs,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CORE_BEHAVIOR_TABLE_SPECS,
)
from fisheye.analytics_exports.validated_behavior_profiles import (
    resolve_validated_behavior_profile,
)
import fisheye.utils.materialize_validated_behavior_bundle_cohort as bundle_cli
import fisheye.utils.materialize_validated_behavior_cohort_export as export_cli

SCIENTIFIC_TABLES = {
    "kinematics_samples",
    "subject_body_frame_samples",
    "eye_trace_samples",
    "tail_trace_samples",
    "canonical_swim_bouts",
}
RUN_PREFIXES = {
    "track_kinematics": "analysis/track_kinematics_runs/offline/",
    "swim_bouts": "analysis/swim_bout_runs/",
    "subject_shape": "analysis/subject_shape_runs/",
    "eye_angles": "analysis/eye_angle_runs/",
    "tail_kinematics": "analysis/tail_kinematics_runs/",
}


def _membership_for_role(role: str) -> dict[str, object]:
    return {
        "members": [
            {
                "recording_id": "recording-a",
                "membership_state": "admitted",
                "admission_receipts": [{"role": role}],
            }
        ]
    }


def _execution_report(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    recording_id = "recording-a"
    zarr_path = (tmp_path / f"{recording_id}_analysis.zarr").resolve()
    temporal_policy = {
        "activity_spatial": {
            "resolution": "fixed_time_bins",
            "bin_size_s": 5.0,
            "source_authority": "framewise_zarr",
        },
        "eye_traces": {
            "resolution": "framewise",
            "source_authority": "framewise_zarr",
        },
        "kinematics": {
            "resolution": "sampled",
            "sample_rate_hz": 10.0,
            "source_authority": "framewise_zarr",
        },
        "tail_traces": {
            "resolution": "framewise",
            "source_authority": "framewise_zarr",
        },
    }
    workflow_nodes = [
        {
            "depends_on": [],
            "description": f"fixture {node_id}",
            "execution_policy": "run_or_reuse",
            "id": node_id,
            "kind": "zarr_stage",
            "output_run_from": None,
            "runnable": True,
            "stage_id": node_id,
            "temporal_product": None,
        }
        for node_id in CORE_BEHAVIOR_REQUIRED_STAGE_NODES
    ]
    workflow = {
        "description": "fixture core behavior",
        "nodes": workflow_nodes,
        "run_selection": {
            node_id: f"{node_id}_run" for node_id in CORE_BEHAVIOR_REQUIRED_STAGE_NODES
        },
        "schema_id": "palette.analysis_workflow",
        "schema_version": 1,
        "targets": ["kinematics_samples", "eye_traces", "tail_traces"],
        "temporal_policy": temporal_policy,
        "workflow_id": "core_behavior_v1",
    }
    plan_nodes = [
        {
            "node_id": node_id,
            "stage_id": node_id,
            "depends_on": [],
            "action": "run",
        }
        for node_id in CORE_BEHAVIOR_REQUIRED_STAGE_NODES
    ]
    output_runs = {
        node_id: f"{node_id}_run" for node_id in CORE_BEHAVIOR_REQUIRED_STAGE_NODES
    }
    report: dict[str, object] = {
        "schema_id": "palette.analysis_workflow_execution",
        "schema_version": 3,
        "execution_id": "core-behavior-fixture",
        "mode": "apply",
        "status": "complete",
        "created_at_utc": "2026-09-03T12:00:00+00:00",
        "completed_at_utc": "2026-09-03T12:01:00+00:00",
        "host": "workstation",
        "lsf_job_id": None,
        "palette_git": {
            "branch": "fixture",
            "commit_hash": "a" * 40,
            "dirty_files": [],
            "is_dirty": False,
            "remote_url": "https://example.invalid/palette.git",
            "short_hash": "a" * 8,
            "top_level": str(tmp_path.resolve()),
        },
        "zarr_path": str(zarr_path),
        "registry_write_mode": "deferred_to_serial_finalizer",
        "workflow": workflow,
        "execution_plan": {
            "schema_id": "palette.analysis_workflow_execution",
            "schema_version": 3,
            "execution_id": "core-behavior-fixture",
            "workflow_plan": {
                "workflow_id": "core_behavior_v1",
                "temporal_policy": temporal_policy,
                "nodes": plan_nodes,
            },
            "output_runs": output_runs,
        },
        "node_results": [
            {
                "node_id": node_id,
                "stage_id": node_id,
                "status": "complete",
                "run_name": output_runs[node_id],
                "verification": {
                    "available": True,
                    "completion_status": "complete",
                    "run_name": output_runs[node_id],
                    "stage_id": node_id,
                    "artifact_path": f"{RUN_PREFIXES[node_id]}{output_runs[node_id]}",
                },
            }
            for node_id in CORE_BEHAVIOR_REQUIRED_STAGE_NODES
        ],
        "error": None,
    }
    return zarr_path, recording_id, report


def test_core_behavior_profile_uses_the_existing_cohort_surface() -> None:
    profile = resolve_validated_behavior_profile(CORE_BEHAVIOR_EXPORT_PROFILE_ID)

    assert profile.table_specs is CORE_BEHAVIOR_TABLE_SPECS
    assert set(validate_table_specs(profile.table_specs)) == (
        set(CORE_TABLE_NAMES) | SCIENTIFIC_TABLES
    )
    assert set(profile.row_extractors()) == SCIENTIFIC_TABLES
    assert set(core_behavior_capability_contract()["keys"]) == set(
        CORE_BEHAVIOR_CAPABILITY_KEYS
    )
    assert all(
        dict(spec.semantic_metadata)["publication_surface"] == "validated_behavior/v1"
        for name, spec in profile.table_specs.items()
        if name in SCIENTIFIC_TABLES
    )

    root = Path("/tmp/palette-validated-behavior-profile-test")
    assert validated_behavior_manifest_path(root, "core-behavior-test") == (
        root
        / "validated_behavior"
        / "v1"
        / "manifests"
        / "export_run_id=core-behavior-test.json"
    )
    assert EXPORT_PLAN_METHOD_ID == "closed_membership_recording_shard_plan_v2"
    assert EXPORT_METHOD_ID == "receipt_composed_manifest_selected_parquet_v2"


def test_completed_execution_report_is_typed_admission_not_name_authority(
    tmp_path: Path,
) -> None:
    zarr_path, recording_id, report = _execution_report(tmp_path)
    validated = validate_core_behavior_execution_report(
        report,
        expected_analysis_zarr=zarr_path,
        expected_recording_id=recording_id,
    )
    assert set(validated["runs"]) == set(CORE_BEHAVIOR_REQUIRED_STAGE_NODES)
    assert validated["recording_id"] == recording_id

    report_path = tmp_path / "execution_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    binding, rebound = bind_core_behavior_execution_report(
        report_path,
        recording_id=recording_id,
        analysis_zarr=zarr_path,
    )
    assert binding["role"] == CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE
    assert binding["record_sha256"] == rebound["record_sha256"]

    report["node_results"][0]["verification"]["artifact_path"] = (  # type: ignore[index]
        "analysis/plausibly_named_but_wrong_parent"
    )
    with pytest.raises(
        ValidatedBehaviorAdmissionError,
        match="artifact path is inexact",
    ):
        validate_core_behavior_execution_report(
            report,
            expected_analysis_zarr=zarr_path,
            expected_recording_id=recording_id,
        )


def test_completed_execution_report_accepts_canonical_framewise_motion(
    tmp_path: Path,
) -> None:
    zarr_path, recording_id, report = _execution_report(tmp_path)
    temporal_policy = report["workflow"]["temporal_policy"]  # type: ignore[index]
    temporal_policy["kinematics"] = {  # type: ignore[index]
        "resolution": "framewise",
        "source_authority": "framewise_zarr",
    }
    report["execution_plan"]["workflow_plan"]["temporal_policy"] = (  # type: ignore[index]
        temporal_policy
    )

    validated = validate_core_behavior_execution_report(
        report,
        expected_analysis_zarr=zarr_path,
        expected_recording_id=recording_id,
    )

    assert validated["temporal_policy"]["kinematics"] == {
        "resolution": "framewise",
        "source_authority": "framewise_zarr",
    }


def test_execution_report_rejects_implicit_or_ambiguous_motion_sampling(
    tmp_path: Path,
) -> None:
    zarr_path, recording_id, report = _execution_report(tmp_path)
    temporal_policy = report["workflow"]["temporal_policy"]  # type: ignore[index]
    temporal_policy["kinematics"].pop("source_authority")  # type: ignore[index]

    with pytest.raises(
        ValidatedBehaviorAdmissionError,
        match="temporal policy is not canonical",
    ):
        validate_core_behavior_execution_report(
            report,
            expected_analysis_zarr=zarr_path,
            expected_recording_id=recording_id,
        )


def test_bundle_cli_dispatches_typed_roles_without_a_second_command() -> None:
    assert (
        bundle_cli._bundle_adapter_for_membership(  # noqa: SLF001
            _membership_for_role(CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE)
        )
        == CORE_BEHAVIOR_BUNDLE_ADAPTER_ID
    )
    assert (
        bundle_cli._bundle_adapter_for_membership(  # noqa: SLF001
            _membership_for_role(EXACT_CHASER_ADMISSION_ROLE)
        )
        == RECORDING_BUNDLE_ADAPTER_ID
    )
    mixed = {
        "members": [
            *_membership_for_role(CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE)["members"],
            {
                "recording_id": "recording-b",
                "membership_state": "admitted",
                "admission_receipts": [{"role": EXACT_CHASER_ADMISSION_ROLE}],
            },
        ]
    }
    with pytest.raises(
        bundle_cli.ValidatedBehaviorCohortCliError,
        match="cannot mix scientific bundle profiles",
    ):
        bundle_cli._bundle_adapter_for_membership(mixed)  # noqa: SLF001


def test_profile_aware_bundle_cannot_be_reinterpreted_by_another_export() -> None:
    bundle_set = {
        "bundle_profile": {"export_profile_id": CORE_BEHAVIOR_EXPORT_PROFILE_ID}
    }
    assert (
        export_core._require_declared_bundle_export_profile(  # noqa: SLF001
            bundle_set, CORE_BEHAVIOR_EXPORT_PROFILE_ID
        )
        == CORE_BEHAVIOR_EXPORT_PROFILE_ID
    )
    with pytest.raises(
        export_core.ValidatedBehaviorExportError,
        match="not requested profile",
    ):
        export_core._require_declared_bundle_export_profile(  # noqa: SLF001
            bundle_set, "validated_recording_behavior_phase_c_v1"
        )

    # Existing Phase-C bundle sets predate this declaration and remain valid.
    assert (
        export_core._require_declared_bundle_export_profile(  # noqa: SLF001
            {"bundle_profile": {"adapter_id": RECORDING_BUNDLE_ADAPTER_ID}},
            "validated_recording_behavior_phase_c_v1",
        )
        == "validated_recording_behavior_phase_c_v1"
    )


def test_core_behavior_finalize_routes_through_the_generic_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = resolve_validated_behavior_profile(CORE_BEHAVIOR_EXPORT_PROFILE_ID)
    plan = {"export_run_id": "core-behavior-test"}
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        export_cli,
        "_read_plan_for_execution",
        lambda _path: (plan, profile),
    )

    def publish(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "manifest_path": str(
                validated_behavior_manifest_path(
                    tmp_path.resolve(), "core-behavior-test"
                )
            ),
            "record_sha256": "a" * 64,
            "schema_version": 2,
            "validation_receipt": {"record_sha256": "b" * 64},
            "transfer_receipt": {"record_sha256": "c" * 64},
            "row_counts_by_table": {},
            "process_telemetry": {
                "policy_id": "validated_behavior_finalize_process_telemetry_v1"
            },
            "safety": {"selector_eligible": False},
        }

    monkeypatch.setattr(export_cli, "publish_validated_behavior_cohort", publish)
    result = export_cli._finalize_command(  # noqa: SLF001
        SimpleNamespace(plan=tmp_path / "plan.json", generation_id="generation-a")
    )

    assert observed == {
        "plan_path": tmp_path / "plan.json",
        "table_specs": CORE_BEHAVIOR_TABLE_SPECS,
        "generation_id": "generation-a",
    }
    assert result["manifest_path"].endswith(
        "/validated_behavior/v1/manifests/export_run_id=core-behavior-test.json"
    )
