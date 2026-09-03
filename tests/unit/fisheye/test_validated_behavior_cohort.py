from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import fisheye.analysis_workflows.validated_behavior_cohort as core
import fisheye.analysis_workflows.validated_behavior_cohort_adapters as adapters
import fisheye.utils.materialize_validated_behavior_bundle_cohort as cohort_cli
from fisheye.analysis_workflows.exact_chaser_projection_receipt import (
    EPOCH_ALIGNMENT_RECEIPT_SCHEMA_VERSION,
    EXACT_CHILD_KEYS_V7,
    POLICY as PROJECTION_POLICY,
    RECEIPT_SCHEMA_ID,
    RECEIPT_STATUS,
    RELATIVE_CHILD_KEYS,
    SAFETY as PROJECTION_SAFETY,
)
from fisheye.analysis_workflows.validated_recording_behavior_bundle import (
    BUNDLE_METHOD_ID,
    BUNDLE_SCHEMA_ID,
    BUNDLE_SCHEMA_VERSION,
    BUNDLE_STATUS,
    CAPABILITY_KEYS,
)
from fisheye.cohorts.registry import (
    MANIFEST_CANONICALIZATION,
    compute_manifest_sha256,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.materialize_composable_chaser_successor_cohort import (
    EXPECTED_SAFETY as TASK_SAFETY,
)

COMMIT = "a" * 40
NOW = "2026-08-31T12:00:00Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _digest(label: str) -> str:
    return canonical_json_sha256({"fixture": label})


def _evidence(
    detail: str | None = None,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    return {
        "evidence_type": "fixture_evidence_v1",
        "detail": detail,
        "path": None if path is None else str(path.resolve()),
        "file_sha256": None if path is None else adapters.sha256_file(path),
        "record_sha256": None,
    }


def _receipt_binding(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": adapters.EXACT_CHASER_ADMISSION_ROLE,
        "path": str(path.resolve()),
        "file_sha256": adapters.sha256_file(path),
        "record_sha256": receipt["record_sha256"],
        "schema_id": receipt["schema_id"],
        "schema_version": receipt["schema_version"],
    }


def _projection_receipt(archive: Path, recording_id: str) -> dict[str, Any]:
    def child(key: str) -> dict[str, str]:
        return {
            "receipt_path": str((archive.parent / f"{key}.json").resolve()),
            "receipt_sha256": _digest(f"receipt:{key}"),
            "run_path": f"analysis/example_runs/{key}",
            "manifest_sha256": _digest(f"manifest:{key}"),
            "payload_digest": _digest(f"payload:{key}"),
        }

    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": EPOCH_ALIGNMENT_RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "analysis_zarr": str(archive.resolve()),
        "recording_id": recording_id,
        "exact_children": {key: child(key) for key in EXACT_CHILD_KEYS_V7},
        "relative_frame_children": {key: child(key) for key in RELATIVE_CHILD_KEYS},
        "policy": PROJECTION_POLICY,
        "safety": PROJECTION_SAFETY,
        "software_authority": {"repository": "palette", "commit": COMMIT},
        "created_at_utc": NOW,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def _policies(tmp_path: Path, count: int) -> tuple[dict[str, Any], ...]:
    incident = tmp_path / "identity.md"
    incident.write_text("operator decision\n", encoding="utf-8")
    unit = adapters.recording_scoped_analysis_unit_policy(
        distinct_animal_count=count,
        decision_timestamp_utc="2026-08-18T12:31:57Z",
        decision_evidence_path=incident,
        decision_evidence_file_sha256=adapters.sha256_file(incident),
        capture_subject_uuid_reuse_count=0,
    )
    temporal = core.policy_envelope(
        {
            "policy_id": "fixture_temporal_policy_v1",
            "temporal_alignment_class": "camera_exposure_time",
            "physical_presentation_verified": True,
        }
    )
    return unit, adapters.missing_acquisition_batch_policy(), temporal


def _generic_membership(tmp_path: Path) -> dict[str, Any]:
    analysis_root = (tmp_path / "recordings").resolve()
    receipt_root = (tmp_path / "receipts").resolve()
    analysis_root.mkdir()
    receipt_root.mkdir()
    source = tmp_path / "source.json"
    _write_json(source, {"members": ["recording-a", "recording-b"]})
    receipt = receipt_root / "admission.json"
    _write_json(receipt, {"receipt": "fixture"})
    source_members = [{"row": 1}, {"row": 2}]
    unit, batch, temporal = _policies(tmp_path, 2)
    members = [
        {
            "source_ordinal": 1,
            "dataset_id": "dataset-a",
            "recording_id": "recording-a",
            "analysis_zarr": str(analysis_root / "recording-a.zarr"),
            "protocol_names": ["behavior-a"],
            "protocol_hashes": [_digest("protocol-a")],
            "source_member_sha256": canonical_json_sha256(source_members[0]),
            "source_subject_ids": ["capture-subject-a"],
            "source_subject_identity_status": "capture_time_non_authoritative",
            "acquisition_batch_id": None,
            "acquisition_batch_identity_status": "missing_historical_not_inferred",
            "analysis_unit_kind": "recording",
            "analysis_unit_id": "recording-a",
            "membership_state": "admitted",
            "reason_code": None,
            "disposition_evidence": _evidence("validated admission"),
            "admission_receipts": [
                {
                    "role": "validated_behavior_receipt",
                    "path": str(receipt),
                    "file_sha256": adapters.sha256_file(receipt),
                    "record_sha256": _digest("admission-record"),
                    "schema_id": "example.validated_behavior_receipt",
                    "schema_version": 1,
                }
            ],
        },
        {
            "source_ordinal": 2,
            "dataset_id": "dataset-b",
            "recording_id": "recording-b",
            "analysis_zarr": str(analysis_root / "recording-b.zarr"),
            "protocol_names": ["behavior-b"],
            "protocol_hashes": [_digest("protocol-b")],
            "source_member_sha256": canonical_json_sha256(source_members[1]),
            "source_subject_ids": [],
            "source_subject_identity_status": "unavailable_in_bound_source_membership",
            "acquisition_batch_id": None,
            "acquisition_batch_identity_status": "missing_historical_not_inferred",
            "analysis_unit_kind": "recording",
            "analysis_unit_id": "recording-b",
            "membership_state": "invalid",
            "reason_code": "invalid_source_authority",
            "disposition_evidence": _evidence("invalid source"),
            "admission_receipts": [],
        },
    ]
    return core.build_validated_behavior_cohort_membership(
        membership_id="mixed-protocol-cohort-v1",
        source_membership={
            "adapter_id": "example_source_adapter_v1",
            "schema_id": "example.frozen_membership",
            "schema_version": 1,
            "profile": "example_profile_v1",
            "path": str(source.resolve()),
            "file_sha256": adapters.sha256_file(source),
            "record_sha256": _digest("source"),
            "member_count": 2,
            "source_members_sha256": canonical_json_sha256(
                [canonical_json_sha256(item) for item in source_members]
            ),
        },
        members=members,
        analysis_zarr_root=analysis_root,
        admission_receipt_root=receipt_root,
        analysis_unit_policy=unit,
        acquisition_batch_policy=batch,
        temporal_alignment_policy=temporal,
        palette_commit=COMMIT,
        created_at_utc=NOW,
    )


def _reseal_membership(value: dict[str, Any]) -> None:
    for member in value["members"]:
        body = {key: item for key, item in member.items() if key != "member_sha256"}
        member["member_sha256"] = canonical_json_sha256(body)
    value["members_sha256"] = canonical_json_sha256(value["members"])
    value["state_counts"] = {
        state: sum(member["membership_state"] == state for member in value["members"])
        for state in core.MEMBERSHIP_STATES
    }
    body = {key: item for key, item in value.items() if key != "record_sha256"}
    value["record_sha256"] = canonical_json_sha256(body)


def test_generic_membership_accepts_multiple_protocols_and_exact_dispositions(
    tmp_path: Path,
) -> None:
    membership = _generic_membership(tmp_path)
    validated = core.validate_validated_behavior_cohort_membership(membership)

    assert validated["member_count"] == 2
    assert validated["state_counts"] == {
        "admitted": 1,
        "excluded": 0,
        "invalid": 1,
        "unavailable": 0,
    }
    assert [member["protocol_names"][0] for member in validated["members"]] == [
        "behavior-a",
        "behavior-b",
    ]


def test_membership_allows_absent_protocol_identity_but_rejects_half_identity(
    tmp_path: Path,
) -> None:
    membership = _generic_membership(tmp_path)
    membership["members"][0]["protocol_names"] = []
    membership["members"][0]["protocol_hashes"] = []
    _reseal_membership(membership)

    validated = core.validate_validated_behavior_cohort_membership(membership)
    assert validated["members"][0]["protocol_names"] == ()
    assert validated["members"][0]["protocol_hashes"] == ()

    membership["members"][0]["protocol_names"] = ["unbound-protocol"]
    _reseal_membership(membership)
    with pytest.raises(
        core.ValidatedBehaviorCohortError,
        match="protocol names and hashes must be coherently present or absent",
    ):
        core.validate_validated_behavior_cohort_membership(membership)


def test_membership_rejects_duplicate_recording_even_when_redigested(
    tmp_path: Path,
) -> None:
    membership = _generic_membership(tmp_path)
    membership["members"][1]["recording_id"] = "recording-a"
    membership["members"][1]["analysis_unit_id"] = "recording-a"
    _reseal_membership(membership)

    with pytest.raises(
        core.ValidatedBehaviorCohortError, match="duplicate recording_id"
    ):
        core.validate_validated_behavior_cohort_membership(membership)


def test_membership_rejects_source_subject_as_analysis_unit(tmp_path: Path) -> None:
    membership = _generic_membership(tmp_path)
    membership["analysis_unit_policy"] = core.policy_envelope(
        {
            "policy_id": "unsafe_subject_unit_v1",
            "analysis_unit_kind": "subject",
            "member_id_field": "recording_id",
        }
    )
    membership["members"][0]["analysis_unit_kind"] = "subject"
    membership["members"][0]["source_subject_ids"] = ["recording-a"]
    membership["members"][1]["analysis_unit_kind"] = "subject"
    _reseal_membership(membership)

    with pytest.raises(
        core.ValidatedBehaviorCohortError, match="source subject identity"
    ):
        core.validate_validated_behavior_cohort_membership(membership)


def test_bundle_set_core_uses_profile_declared_capabilities_not_chaser_formulas(
    tmp_path: Path,
) -> None:
    membership = _generic_membership(tmp_path)
    membership_path = tmp_path / "membership.json"
    _write_json(membership_path, membership)
    bundle_root = (tmp_path / "bundles").resolve()
    bundle_root.mkdir()
    bundle_path = bundle_root / "recording-a.json"
    _write_json(bundle_path, {"bundle": "example"})
    contract = core.build_capability_contract(
        profile_id="feeding_behavior_v1",
        keys=("feeding_latency", "swim_bout_rate"),
        reason_codes_by_state={
            "complete": (None,),
            "inapplicable": ("not_requested",),
            "invalid": ("invalid_source",),
            "review_required": ("review_not_accepted",),
            "stale": ("stale_source",),
            "unavailable": ("blocked_by_invalid_membership",),
        },
    )
    complete_capabilities = {
        key: {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": {"scope": "scientific_child", "key": key},
        }
        for key in contract["keys"]
    }
    invalid_capabilities = {
        key: {
            "state": "unavailable",
            "reason_code": "blocked_by_invalid_membership",
            "detail": "source invalid",
            "binding": None,
        }
        for key in contract["keys"]
    }
    bundle_set = core.build_validated_behavior_bundle_set(
        bundle_set_id="generic-feeding-bundles-v1",
        membership=membership,
        membership_path=membership_path,
        membership_file_sha256=adapters.sha256_file(membership_path),
        bundle_root=bundle_root,
        bundle_profile={"adapter_id": "feeding_bundle_v1"},
        capability_contract=contract,
        members=[
            {
                "recording_id": "recording-a",
                "bundle_state": "complete",
                "reason_code": None,
                "bundle": {
                    "adapter_id": "feeding_bundle_v1",
                    "path": str(bundle_path),
                    "file_sha256": adapters.sha256_file(bundle_path),
                    "record_sha256": _digest("feeding-bundle"),
                    "schema_id": "example.feeding_bundle",
                    "schema_version": 1,
                    "method_id": "feeding_bundle_v1",
                    "status": "complete",
                    "receipt_bindings": [
                        {
                            "role": "feeding_observation_receipt",
                            "path": str(
                                (tmp_path / "receipts" / "admission.json").resolve()
                            ),
                            "file_sha256": adapters.sha256_file(
                                tmp_path / "receipts" / "admission.json"
                            ),
                            "record_sha256": _digest("admission-record"),
                            "schema_id": "example.validated_behavior_receipt",
                            "schema_version": 1,
                        }
                    ],
                    "binding_inventory_sha256": _digest("feeding-bindings"),
                },
                "capabilities": complete_capabilities,
            },
            {
                "recording_id": "recording-b",
                "bundle_state": "invalid",
                "reason_code": "invalid_source_authority",
                "bundle": None,
                "capabilities": invalid_capabilities,
            },
        ],
        palette_commit=COMMIT,
        created_at_utc=NOW,
    )

    validated = core.validate_validated_behavior_bundle_set(
        bundle_set, membership=membership
    )
    assert validated["capability_contract"]["keys"] == (
        "feeding_latency",
        "swim_bout_rate",
    )
    assert validated["state_counts"]["complete"] == 1


def _task_entry(
    *, analysis_root: Path, recording_id: str, index: int
) -> dict[str, Any]:
    return {
        "task_index": index,
        "dataset_id": f"dataset-{index}",
        "recording_id": recording_id,
        "analysis_zarr": str((analysis_root / f"{recording_id}.zarr").resolve()),
        "protocol_name": "goodbatbadbat",
        "protocol_hash": _digest("protocol"),
    }


def _task(path: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    body = {
        "schema_id": "palette.composable_chaser_successor_cohort_task",
        "schema_version": 5,
        "created_at_utc": NOW,
        "operations_root": str(path.parent.resolve()),
        "recording_count": len(entries),
        "runnable_task_indices": list(range(1, len(entries) + 1)),
        "safety": TASK_SAFETY,
        "selection_policy": {},
        "source_registry_snapshot": {
            "path": "/tmp/historical-registry.json",
            "row_count": len(entries),
            "sha256": _digest("registry"),
        },
        "status_counts": {"resume": len(entries)},
        "entries": entries,
    }
    return {**body, "task_sha256": canonical_json_sha256(body)}


def _historical_membership_fixture(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    analysis_root = (tmp_path / "recordings").resolve()
    receipt_root = (tmp_path / "operations").resolve()
    analysis_root.mkdir()
    receipt_root.mkdir()
    entries = [
        _task_entry(analysis_root=analysis_root, recording_id="recording-a", index=1),
        _task_entry(analysis_root=analysis_root, recording_id="recording-b", index=2),
    ]
    task_path = tmp_path / "task.json"
    _write_json(task_path, _task(task_path, entries))
    projection = _projection_receipt(Path(entries[0]["analysis_zarr"]), "recording-a")
    receipt_path = receipt_root / "recording-a" / "projection.v7.json"
    _write_json(receipt_path, projection)
    dispositions = {
        "recording-a": {
            "membership_state": "admitted",
            "reason_code": None,
            "disposition_evidence": _evidence("exact projection"),
            "admission_receipts": [_receipt_binding(receipt_path, projection)],
        },
        "recording-b": {
            "membership_state": "invalid",
            "reason_code": "invalid_semantic_selection",
            "disposition_evidence": _evidence(
                "raw semantic step bounds overlap or are not strictly ordered"
            ),
            "admission_receipts": [],
        },
    }
    unit, batch, _ = _policies(tmp_path, 2)
    temporal = adapters.historical_controller_proxy_temporal_policy()
    membership = adapters.build_membership_from_composable_chaser_task_v5(
        task_path,
        membership_id="historical-chaser-import-v1",
        dispositions_by_recording=dispositions,
        analysis_zarr_root=analysis_root,
        admission_receipt_root=receipt_root,
        analysis_unit_policy=unit,
        acquisition_batch_policy=batch,
        temporal_alignment_policy=temporal,
        palette_commit=COMMIT,
        created_at_utc=NOW,
    )
    return membership, task_path


def test_historical_adapter_imports_then_revalidates_exact_parent_roster(
    tmp_path: Path,
) -> None:
    membership, _ = _historical_membership_fixture(tmp_path)

    validated = adapters.validate_membership_current_sources(membership)

    assert validated["source_membership"]["adapter_id"] == (
        adapters.HISTORICAL_TASK_ADAPTER_ID
    )
    assert validated["state_counts"]["admitted"] == 1
    assert validated["state_counts"]["invalid"] == 1


def test_historical_adapter_rejects_changed_task_with_old_digest(
    tmp_path: Path,
) -> None:
    membership, task_path = _historical_membership_fixture(tmp_path)
    changed = json.loads(task_path.read_text(encoding="utf-8"))
    changed["entries"][0]["dataset_id"] = "substituted-dataset"
    _write_json(task_path, changed)

    with pytest.raises(
        core.ValidatedBehaviorCohortError, match="source-membership file"
    ):
        adapters.validate_membership_current_sources(membership)


def test_historical_adapter_requires_disposition_for_every_parent_member(
    tmp_path: Path,
) -> None:
    analysis_root = (tmp_path / "recordings").resolve()
    receipt_root = (tmp_path / "operations").resolve()
    analysis_root.mkdir()
    receipt_root.mkdir()
    task_path = tmp_path / "task.json"
    entries = [
        _task_entry(analysis_root=analysis_root, recording_id="recording-a", index=1),
        _task_entry(analysis_root=analysis_root, recording_id="recording-b", index=2),
    ]
    _write_json(task_path, _task(task_path, entries))
    unit, batch, temporal = _policies(tmp_path, 2)

    with pytest.raises(
        core.ValidatedBehaviorCohortError, match="missing=.*recording-b"
    ):
        adapters.build_membership_from_composable_chaser_task_v5(
            task_path,
            membership_id="incomplete",
            dispositions_by_recording={
                "recording-a": {
                    "membership_state": "invalid",
                    "reason_code": "invalid_source_authority",
                    "disposition_evidence": _evidence("invalid"),
                    "admission_receipts": [],
                }
            },
            analysis_zarr_root=analysis_root,
            admission_receipt_root=receipt_root,
            analysis_unit_policy=unit,
            acquisition_batch_policy=batch,
            temporal_alignment_policy=temporal,
            palette_commit=COMMIT,
            created_at_utc=NOW,
        )


def test_frozen_cohort_v2_imports_through_same_membership_interface(
    tmp_path: Path,
) -> None:
    analysis_root = (tmp_path / "recordings").resolve()
    receipt_root = (tmp_path / "operations").resolve()
    analysis_root.mkdir()
    receipt_root.mkdir()
    recording_id = "recording-frozen"
    archive = analysis_root / f"{recording_id}.zarr"
    projection = _projection_receipt(archive, recording_id)
    receipt_path = receipt_root / recording_id / "projection.v7.json"
    _write_json(receipt_path, projection)
    member = {
        "dataset_id": "dataset-frozen",
        "recording_id": recording_id,
        "zarr_path": str(archive),
        "zarr_origin": "source",
        "zarr_use": "analysis",
        "dataset_status": "active",
        "protocol_names": ["another-protocol"],
        "protocol_hashes": [_digest("another-protocol")],
        "subject_values": {"subject_uuid": ["00000000-0000-4000-8000-000000000001"]},
    }
    cohort = {
        "schema_id": "palette.frozen_cohort_manifest",
        "schema_version": 2,
        "manifest_canonicalization": MANIFEST_CANONICALIZATION,
        "created_utc": NOW,
        "cohort_id": "frozen",
        "cohort_name": "Frozen",
        "purpose": "fixture",
        "cohort_query": {"schema_id": "fixture"},
        "cohort_query_sha256": canonical_json_sha256({"schema_id": "fixture"}),
        "registry": {
            "query_snapshot_sha256": _digest("snapshot"),
            "access_mode": "read_only",
            "registry_uuid": "00000000-0000-4000-8000-000000000002",
            "identity_provenance": "schema_managed",
            "identity_minted_at_utc": NOW,
        },
        "selection_policy": {"include_every_match": True, "limit": None},
        "member_count": 1,
        "members": [member],
        "selection_summary": {"included_count": 1, "blocked_count": 0},
    }
    cohort["manifest_sha256"] = compute_manifest_sha256(cohort)
    cohort_path = tmp_path / "frozen.json"
    _write_json(cohort_path, cohort)
    unit, batch, temporal = _policies(tmp_path, 1)
    membership = adapters.build_membership_from_frozen_cohort_v2(
        cohort_path,
        membership_id="future-frozen-import-v1",
        dispositions_by_recording={
            recording_id: {
                "membership_state": "admitted",
                "reason_code": None,
                "disposition_evidence": _evidence("exact projection"),
                "admission_receipts": [_receipt_binding(receipt_path, projection)],
            }
        },
        analysis_zarr_root=analysis_root,
        admission_receipt_root=receipt_root,
        analysis_unit_policy=unit,
        acquisition_batch_policy=batch,
        temporal_alignment_policy=temporal,
        palette_commit=COMMIT,
        created_at_utc=NOW,
    )

    validated = adapters.validate_membership_current_sources(membership)
    assert validated["source_membership"]["adapter_id"] == (
        adapters.FROZEN_COHORT_ADAPTER_ID
    )
    assert validated["members"][0]["source_subject_ids"] == (
        "00000000-0000-4000-8000-000000000001",
    )
    assert validated["members"][0]["analysis_unit_id"] == recording_id


def test_recording_bundle_adapter_preserves_profile_capabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    membership, _ = _historical_membership_fixture(tmp_path)
    membership_path = tmp_path / "membership.json"
    _write_json(membership_path, membership)
    bundle_root = (tmp_path / "bundles").resolve()
    bundle_root.mkdir()
    bundle_path = bundle_root / "recording-a.json"
    _write_json(bundle_path, {})
    admission = membership["members"][0]["admission_receipts"][0]
    capabilities = {
        key: {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding_scope": "source_bindings",
            "binding_key": key,
        }
        for key in CAPABILITY_KEYS
    }
    fake_bundle = {
        "schema_id": BUNDLE_SCHEMA_ID,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "method_id": BUNDLE_METHOD_ID,
        "status": BUNDLE_STATUS,
        "analysis_zarr": membership["members"][0]["analysis_zarr"],
        "recording_id": "recording-a",
        "projection_receipt": {
            "receipt_path": admission["path"],
            "receipt_sha256": admission["record_sha256"],
            "schema_id": admission["schema_id"],
            "schema_version": admission["schema_version"],
        },
        "source_bindings": {"source": {"fixture": True}},
        "scientific_child_bindings": {"child": {"fixture": True}},
        "capabilities": capabilities,
        "record_sha256": _digest("bundle-record"),
    }
    monkeypatch.setattr(
        adapters,
        "read_validated_recording_behavior_bundle",
        lambda *_args, **_kwargs: deepcopy(fake_bundle),
    )

    bundle_set = adapters.build_bundle_set_from_validated_recording_behavior_bundles(
        bundle_set_id="recording-bundle-set-v1",
        membership=membership,
        membership_path=membership_path,
        bundle_paths_by_recording={"recording-a": bundle_path},
        bundle_root=bundle_root,
        palette_commit=COMMIT,
        created_at_utc=NOW,
        validate_current_sources=False,
    )

    assert bundle_set["state_counts"] == {
        "complete": 1,
        "excluded": 0,
        "invalid": 1,
        "unavailable": 0,
    }
    invalid = bundle_set["members"][1]
    assert invalid["bundle"] is None
    assert invalid["capabilities"]["semantic_epochs"]["state"] == "invalid"
    assert invalid["capabilities"]["provider_motion"]["reason_code"] == (
        "blocked_by_invalid_membership"
    )


def _invalid_dispositions_document(
    task: dict[str, Any], recording_id: str
) -> dict[str, Any]:
    body = {
        "schema_id": adapters.INVALID_DISPOSITIONS_SCHEMA_ID,
        "schema_version": adapters.INVALID_DISPOSITIONS_SCHEMA_VERSION,
        "source_membership_record_sha256": task["task_sha256"],
        "entry_count": 1,
        "entries": [
            {
                "recording_id": recording_id,
                "reason_code": "invalid_semantic_selection",
                "detail": (
                    "raw semantic step bounds overlap or are not strictly ordered"
                ),
            }
        ],
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def test_historical_disposition_planner_uses_explicit_invalid_roster(
    tmp_path: Path,
) -> None:
    analysis_root = (tmp_path / "recordings").resolve()
    operations_root = (tmp_path / "operations").resolve()
    analysis_root.mkdir()
    operations_root.mkdir()
    generation = "receipt-generation-a"
    filename = "projection.v7.json"
    entries = [
        _task_entry(analysis_root=analysis_root, recording_id="recording-a", index=1),
        _task_entry(analysis_root=analysis_root, recording_id="recording-b", index=2),
    ]
    for entry in entries:
        entry["plot_output_dir"] = str(
            (operations_root / entry["recording_id"]).resolve()
        )
    task_path = tmp_path / "task.json"
    task = _task(task_path, entries)
    _write_json(task_path, task)
    receipt_path = (
        operations_root
        / "recording-a"
        / "source_validation_receipts"
        / generation
        / filename
    )
    receipt = _projection_receipt(Path(entries[0]["analysis_zarr"]), "recording-a")
    _write_json(receipt_path, receipt)
    invalid_path = tmp_path / "invalid.json"
    _write_json(invalid_path, _invalid_dispositions_document(task, "recording-b"))

    dispositions = adapters.plan_composable_chaser_task_v5_dispositions(
        task_path,
        receipt_generation=generation,
        receipt_filename=filename,
        invalid_dispositions_path=invalid_path,
    )

    assert dispositions["recording-a"]["membership_state"] == "admitted"
    assert dispositions["recording-b"]["membership_state"] == "invalid"
    assert dispositions["recording-b"]["disposition_evidence"]["path"] == str(
        invalid_path.resolve()
    )


def test_membership_cli_writes_bounded_generic_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_root = (tmp_path / "recordings").resolve()
    operations_root = (tmp_path / "operations").resolve()
    analysis_root.mkdir()
    operations_root.mkdir()
    generation = "receipt-generation-a"
    filename = "projection.v7.json"
    entries = [
        _task_entry(analysis_root=analysis_root, recording_id="recording-a", index=1),
        _task_entry(analysis_root=analysis_root, recording_id="recording-b", index=2),
    ]
    for entry in entries:
        entry["plot_output_dir"] = str(
            (operations_root / entry["recording_id"]).resolve()
        )
    task_path = tmp_path / "task.json"
    task = _task(task_path, entries)
    _write_json(task_path, task)
    receipt = _projection_receipt(Path(entries[0]["analysis_zarr"]), "recording-a")
    _write_json(
        operations_root
        / "recording-a"
        / "source_validation_receipts"
        / generation
        / filename,
        receipt,
    )
    invalid_path = tmp_path / "invalid.json"
    _write_json(invalid_path, _invalid_dispositions_document(task, "recording-b"))
    incident = tmp_path / "incident.md"
    incident.write_text("recording scoped decision\n", encoding="utf-8")
    temporal = tmp_path / "temporal.json"
    _write_json(
        temporal,
        adapters.HISTORICAL_CONTROLLER_PROXY_TEMPORAL_POLICY,
    )
    output = tmp_path / "membership.json"
    monkeypatch.setattr(cohort_cli, "_current_palette_git_state", lambda: (COMMIT, ""))

    assert (
        cohort_cli.main(
            [
                "membership-from-chaser-task-v5",
                "--source-membership",
                str(task_path),
                "--receipt-generation",
                generation,
                "--receipt-filename",
                filename,
                "--invalid-dispositions-json",
                str(invalid_path),
                "--membership-id",
                "fixture-membership-v1",
                "--analysis-zarr-root",
                str(analysis_root),
                "--admission-receipt-root",
                str(operations_root),
                "--identity-decision-evidence",
                str(incident),
                "--identity-decision-timestamp-utc",
                "2026-08-18T12:31:57Z",
                "--distinct-animal-count",
                "2",
                "--capture-subject-uuid-reuse-count",
                "0",
                "--temporal-alignment-policy-json",
                str(temporal),
                "--palette-commit",
                COMMIT,
                "--output-json",
                str(output),
                "--expected-parent-count",
                "2",
                "--expected-admitted-count",
                "1",
                "--expected-invalid-count",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)

    assert summary["state_counts"]["admitted"] == 1
    assert summary["state_counts"]["invalid"] == 1
    assert summary["selector_eligible"] is False
    assert "members" not in summary
    assert output.is_file()


def test_cli_rejects_false_or_dirty_software_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cohort_cli, "_current_palette_git_state", lambda: (COMMIT, ""))
    with pytest.raises(
        cohort_cli.ValidatedBehaviorCohortCliError,
        match="must equal the exact commit",
    ):
        cohort_cli._require_current_software_authority("b" * 40)

    monkeypatch.setattr(
        cohort_cli,
        "_current_palette_git_state",
        lambda: (COMMIT, " M src/fisheye/example.py"),
    )
    with pytest.raises(
        cohort_cli.ValidatedBehaviorCohortCliError,
        match="clean commit-pinned",
    ):
        cohort_cli._require_current_software_authority(COMMIT)
