from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fisheye.analysis_workflows.core_behavior_cohort_adapter import (
    CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
    core_behavior_capability_contract,
)
from fisheye.analysis_workflows.validated_behavior_cohort import (
    build_validated_behavior_bundle_set,
    build_validated_behavior_cohort_membership,
    policy_envelope,
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
    build_validated_behavior_export_plan,
    publish_validated_behavior_cohort,
    validated_behavior_manifest_path,
    write_validated_behavior_export_plan,
    write_validated_behavior_recording_shard,
)
import fisheye.analytics_exports.validated_behavior_cohort as export_core
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_TABLE_NAMES,
    validate_table_specs,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_CAPABILITY_PROFILE_ID,
    CORE_BEHAVIOR_CAPABILITY_PROFILE_ID_V1,
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1,
    CORE_BEHAVIOR_TABLE_SPECS,
    CORE_BEHAVIOR_TABLE_SPECS_V1,
    KINEMATICS_SAMPLES,
    KINEMATICS_SAMPLES_V1,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.analytics_exports.validated_behavior_profiles import (
    ValidatedBehaviorExportProfile,
    ValidatedBehaviorProfileError,
    _validated_profile_map,
    resolve_validated_behavior_profile,
)
from fisheye.analytics_exports.validated_behavior_phase_b_contracts import (
    PHASE_B_TABLE_SPECS,
)
from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
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


def test_core_motion_successor_preserves_v1_profile_and_uses_explicit_semantics() -> (
    None
):
    current = resolve_validated_behavior_profile(CORE_BEHAVIOR_EXPORT_PROFILE_ID)
    v1 = resolve_validated_behavior_profile(CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1)

    assert current.table_specs is CORE_BEHAVIOR_TABLE_SPECS
    assert v1.table_specs is CORE_BEHAVIOR_TABLE_SPECS_V1
    assert KINEMATICS_SAMPLES.schema_version == 2
    assert KINEMATICS_SAMPLES_V1.schema_version == 1
    current_fields = {field.name for field in KINEMATICS_SAMPLES.fields}
    v1_fields = {field.name for field in KINEMATICS_SAMPLES_V1.fields}
    assert {
        "delta_frames",
        "delta_seconds",
        "speed_filtered_mm_s",
        "speed_smoothed_mm_s",
        "frame_path_distance_filtered_mm",
        "frame_path_distance_smoothed_mm",
        "signed_tangential_acceleration_mm_s2",
        "smoothed_signed_tangential_acceleration_mm_s2",
        "cumulative_smoothed_path_distance_mm",
    }.issubset(current_fields)
    assert "acceleration_mm_s2" not in current_fields
    assert "signed_tangential_acceleration_mm_s2" not in v1_fields
    assert "speed_mm_s" in v1_fields
    assert KINEMATICS_SAMPLES_V1.payload_sha256 == (
        "35f7b95cc2c46253365a8ae91dc88b886f2cde7b93f4a1e4a1581bd18be69505"
    )
    assert (
        canonical_json_sha256(export_core._spec_records(CORE_BEHAVIOR_TABLE_SPECS_V1))
        == "bbb9ababe1000bca3d19d1f4cca18403a422c20a019fd9b288dc52b89fc3b98d"
    )


def test_core_motion_successor_versions_capability_contract_with_export_profile() -> (
    None
):
    current = core_behavior_capability_contract(CORE_BEHAVIOR_EXPORT_PROFILE_ID)
    v1 = core_behavior_capability_contract(CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1)

    assert current["profile_id"] == CORE_BEHAVIOR_CAPABILITY_PROFILE_ID
    assert v1["profile_id"] == CORE_BEHAVIOR_CAPABILITY_PROFILE_ID_V1
    assert current["record_sha256"] != v1["record_sha256"]
    assert current["keys"] == v1["keys"]


def test_installed_profile_gate_rejects_competing_motion_projections() -> None:
    invalid_specs = {
        **CORE_BEHAVIOR_TABLE_SPECS,
        "provider_motion_samples": PHASE_B_TABLE_SPECS["provider_motion_samples"],
    }
    invalid = ValidatedBehaviorExportProfile(
        profile_id="invalid_competing_motion",
        table_specs=invalid_specs,
        row_extractor_factory=lambda: {},
    )
    with pytest.raises(
        ValidatedBehaviorProfileError,
        match="competing core-motion projections",
    ):
        _validated_profile_map({invalid.profile_id: invalid})


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_digest(label: str) -> str:
    return canonical_json_sha256({"fixture": label})


def _core_profile_plan(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    commit = "a" * 40
    now = "2026-09-04T12:00:00Z"
    analysis_root = (tmp_path / "recordings").resolve()
    receipt_root = (tmp_path / "receipts").resolve()
    bundle_root = (tmp_path / "bundles").resolve()
    for path in (analysis_root, receipt_root, bundle_root):
        path.mkdir()

    source_member = {"recording_id": "recording-a"}
    source_path = tmp_path / "source-membership.json"
    _write_json(source_path, {"members": [source_member]})
    receipt_path = receipt_root / "recording-a" / "execution-report.json"
    _write_json(receipt_path, {"status": "complete"})
    decision_path = tmp_path / "identity-decision.md"
    decision_path.write_text("one recording is one analysis unit\n", encoding="utf-8")

    membership = build_validated_behavior_cohort_membership(
        membership_id="core-motion-boundary-v2",
        source_membership={
            "adapter_id": "core_motion_boundary_fixture_v1",
            "schema_id": "fixture.closed_membership",
            "schema_version": 1,
            "profile": "core_motion_boundary_fixture_v1",
            "path": str(source_path.resolve()),
            "file_sha256": _file_sha256(source_path),
            "record_sha256": _fixture_digest("source-membership"),
            "member_count": 1,
            "source_members_sha256": canonical_json_sha256(
                [canonical_json_sha256(source_member)]
            ),
        },
        members=[
            {
                "source_ordinal": 1,
                "dataset_id": "dataset-a",
                "recording_id": "recording-a",
                "analysis_zarr": str(analysis_root / "recording-a.zarr"),
                "protocol_names": ["core-behavior"],
                "protocol_hashes": [_fixture_digest("protocol")],
                "source_member_sha256": canonical_json_sha256(source_member),
                "source_subject_ids": ["capture-subject-a"],
                "source_subject_identity_status": "capture_time_non_authoritative",
                "acquisition_batch_id": None,
                "acquisition_batch_identity_status": (
                    "missing_historical_not_inferred"
                ),
                "analysis_unit_kind": "recording",
                "analysis_unit_id": "recording-a",
                "membership_state": "admitted",
                "reason_code": None,
                "disposition_evidence": {
                    "evidence_type": "fixture_decision_v1",
                    "detail": "exact fixture authority accepted",
                    "path": None,
                    "file_sha256": None,
                    "record_sha256": None,
                },
                "admission_receipts": [
                    {
                        "role": CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
                        "path": str(receipt_path.resolve()),
                        "file_sha256": _file_sha256(receipt_path),
                        "record_sha256": _fixture_digest("execution-report"),
                        "schema_id": "fixture.execution_report",
                        "schema_version": 1,
                    }
                ],
            }
        ],
        analysis_zarr_root=analysis_root,
        admission_receipt_root=receipt_root,
        analysis_unit_policy=policy_envelope(
            {
                "policy_id": "recording_scoped_fixture_v1",
                "analysis_unit_kind": "recording",
                "member_id_field": "recording_id",
                "decision_evidence": {
                    "path": str(decision_path.resolve()),
                    "file_sha256": _file_sha256(decision_path),
                },
            }
        ),
        acquisition_batch_policy=policy_envelope(
            {
                "policy_id": "missing_batch_fixture_v1",
                "missing_identity_status": "missing_historical_not_inferred",
                "authoritative_identity_status": "authoritative",
                "inference_allowed": False,
            }
        ),
        temporal_alignment_policy=policy_envelope(
            {
                "policy_id": "framewise_fixture_v1",
                "temporal_alignment_requirement": "exact_acquisition_frame",
            }
        ),
        palette_commit=commit,
        created_at_utc=now,
    )
    membership_path = tmp_path / "membership.json"
    _write_json(membership_path, membership)

    bundle_path = bundle_root / "recording-a.bundle.json"
    _write_json(bundle_path, {"status": "complete"})
    capability_contract = core_behavior_capability_contract()
    capabilities = {
        key: {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": {"fixture_binding": key},
        }
        for key in capability_contract["keys"]
    }
    receipt_binding = membership["members"][0]["admission_receipts"][0]
    bundle_set = build_validated_behavior_bundle_set(
        bundle_set_id="core-motion-boundary-bundles-v2",
        membership=membership,
        membership_path=membership_path,
        membership_file_sha256=_file_sha256(membership_path),
        bundle_root=bundle_root,
        bundle_profile={
            "adapter_id": CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
            "export_profile_id": CORE_BEHAVIOR_EXPORT_PROFILE_ID,
        },
        capability_contract=capability_contract,
        members=[
            {
                "recording_id": "recording-a",
                "bundle_state": "complete",
                "reason_code": None,
                "bundle": {
                    "adapter_id": CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
                    "path": str(bundle_path.resolve()),
                    "file_sha256": _file_sha256(bundle_path),
                    "record_sha256": _fixture_digest("bundle"),
                    "schema_id": "fixture.core_behavior_bundle",
                    "schema_version": 1,
                    "method_id": "fixture_core_behavior_bundle_v1",
                    "status": "complete",
                    "receipt_bindings": [receipt_binding],
                    "binding_inventory_sha256": _fixture_digest("binding-inventory"),
                },
                "capabilities": capabilities,
            }
        ],
        palette_commit=commit,
        created_at_utc=now,
    )
    bundle_set_path = tmp_path / "bundle-set.json"
    _write_json(bundle_set_path, bundle_set)
    plan = build_validated_behavior_export_plan(
        membership_path=membership_path,
        bundle_set_path=bundle_set_path,
        export_run_id="core-motion-writer-reader-boundary-v2",
        shard_root=(tmp_path / "shards").resolve(),
        publication_root=(tmp_path / "publication").resolve(),
        palette_commit=commit,
        palette_repo=Path(__file__).resolve().parents[3],
        table_specs=CORE_BEHAVIOR_TABLE_SPECS,
        export_profile_id=CORE_BEHAVIOR_EXPORT_PROFILE_ID,
        created_at_utc=now,
    )
    plan_path = tmp_path / "plan.json"
    write_validated_behavior_export_plan(plan_path, plan)
    return plan_path, plan


def _fixture_scientific_row(
    *,
    table_name: str,
    plan: dict[str, Any],
    member: dict[str, Any],
    bundle_member: dict[str, Any],
) -> dict[str, Any]:
    spec = CORE_BEHAVIOR_TABLE_SPECS[table_name]
    known: dict[str, Any] = {
        "export_run_id": plan["export_run_id"],
        "recording_id": member["recording_id"],
        "membership_member_sha256": member["member_sha256"],
        "bundle_set_member_sha256": bundle_member["member_sha256"],
        "bundle_record_sha256": bundle_member["bundle"]["record_sha256"],
        "cross_grain_join_authority_sha256": _fixture_digest("join"),
        "source_track_kinematics_scope": "offline",
        "motion_projection_profile_id": "core_motion_physical_v2",
        "acceleration_source_speed_level": "speed_smoothed",
        "cumulative_path_distance_source_level": "smoothed",
        "sampling_stride_frames": 1,
        "source_sample_rate_hz": 30.0,
        "requested_sample_rate_hz": 30.0,
        "nominal_sample_rate_hz": 30.0,
        "position_coordinate_space": "physical_mm",
        "speed_filtered_mm_s": 4.0,
        "speed_smoothed_mm_s": 3.5,
        "signed_tangential_acceleration_mm_s2": -2.0,
        "smoothed_signed_tangential_acceleration_mm_s2": -1.5,
        "cumulative_smoothed_path_distance_mm": 9.5,
    }
    row: dict[str, Any] = {}
    for field in spec.contract.fields:
        if field.name in known:
            row[field.name] = known[field.name]
        elif field.nullable:
            row[field.name] = None
        elif field.arrow_type == "string":
            row[field.name] = (
                _fixture_digest(field.name)
                if any(token in field.name for token in ("sha256", "digest", "hash"))
                else "fixture"
            )
        elif field.arrow_type == "bool":
            row[field.name] = True
        elif field.arrow_type.startswith(("int", "uint")):
            row[field.name] = 0
        elif field.arrow_type.startswith("float"):
            row[field.name] = 0.0
        else:  # pragma: no cover - closed current scientific schemas
            raise AssertionError(f"Unsupported fixture Arrow type: {field.arrow_type}")
    return row


def test_core_motion_v2_real_writer_publisher_unpatched_reader_round_trip(
    tmp_path: Path,
) -> None:
    plan_path, plan = _core_profile_plan(tmp_path)

    def extractor(table_name: str) -> Any:
        def rows(
            plan_value: dict[str, Any],
            member: dict[str, Any],
            bundle_member: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], None]:
            return [
                _fixture_scientific_row(
                    table_name=table_name,
                    plan=plan_value,
                    member=member,
                    bundle_member=bundle_member,
                )
            ], None

        return rows

    extractors = {table_name: extractor(table_name) for table_name in SCIENTIFIC_TABLES}
    write_validated_behavior_recording_shard(
        plan_path=plan_path,
        member_ordinal=1,
        table_specs=CORE_BEHAVIOR_TABLE_SPECS,
        row_extractors=extractors,
        created_at_utc="2026-09-04T12:01:00Z",
    )
    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        table_specs=CORE_BEHAVIOR_TABLE_SPECS,
        generation_id="core-motion-boundary-generation-v2",
        created_at_utc="2026-09-04T12:02:00Z",
    )

    # No table specs are injected here: the reader must resolve the sealed
    # installed profile ID and its v2 kinematics contract itself.
    dataset = ValidatedBehaviorExportDataset.open(
        plan["publication_root"], plan["export_run_id"]
    )
    row = dataset.table("kinematics_samples").collect_bounded(max_rows=2).to_dicts()[0]

    assert published["status"] == "complete_selector_ineligible"
    assert dataset.manifest["export_profile"]["profile_id"] == (
        CORE_BEHAVIOR_EXPORT_PROFILE_ID
    )
    assert dataset.table_specs["kinematics_samples"].contract.schema_version == 2
    assert "provider_motion_samples" not in dataset.table_names
    assert row["speed_filtered_mm_s"] == pytest.approx(4.0)
    assert row["signed_tangential_acceleration_mm_s2"] == pytest.approx(-2.0)
    assert row["cumulative_smoothed_path_distance_mm"] == pytest.approx(9.5)
