from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import polars as pl
import pytest

from fisheye.analysis_workflows.validated_behavior_cohort import (
    build_capability_contract,
    build_validated_behavior_bundle_set,
    build_validated_behavior_cohort_membership,
    policy_envelope,
)
from fisheye.analytics_exports.validated_behavior_cohort import (
    ValidatedBehaviorBatchSource,
    ValidatedBehaviorExportError,
    build_validated_behavior_export_plan,
    planned_shard_receipt_path,
    publish_validated_behavior_cohort,
    read_validated_behavior_export_manifest,
    write_validated_behavior_export_plan,
    write_validated_behavior_recording_shard,
)
from fisheye.analytics_exports.arrow_contract_core import ArrowTableContract, field
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_TABLE_NAMES,
    CORE_TABLE_SPECS,
    ValidatedBehaviorTableSpec,
)
from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
import fisheye.utils.materialize_validated_behavior_cohort_export as export_cli
import fisheye.analytics_exports.validated_behavior_cohort as cohort_export

COMMIT = "a" * 40
NOW = "2026-08-31T12:00:00Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(label: str) -> str:
    return canonical_json_sha256({"fixture": label})


def _membership_and_bundle_set(tmp_path: Path) -> tuple[Path, Path]:
    analysis_root = (tmp_path / "recordings").resolve()
    receipt_root = (tmp_path / "receipts").resolve()
    bundle_root = (tmp_path / "bundles").resolve()
    analysis_root.mkdir()
    receipt_root.mkdir()
    bundle_root.mkdir()
    source_members = [{"source": "recording-a"}, {"source": "recording-b"}]
    source_path = tmp_path / "source-membership.json"
    _write_json(source_path, {"members": source_members})
    admission_path = receipt_root / "recording-a" / "admission.json"
    _write_json(admission_path, {"receipt": "exact-a"})
    decision_path = tmp_path / "identity-decision.md"
    decision_path.write_text("recordings are distinct animals\n", encoding="utf-8")
    unit_policy = policy_envelope(
        {
            "policy_id": "recording_scoped_fixture_v1",
            "analysis_unit_kind": "recording",
            "member_id_field": "recording_id",
            "decision_evidence": {
                "path": str(decision_path.resolve()),
                "file_sha256": _sha256_file(decision_path),
            },
        }
    )
    batch_policy = policy_envelope(
        {
            "policy_id": "missing_batch_fixture_v1",
            "missing_identity_status": "missing_historical_not_inferred",
            "authoritative_identity_status": "authoritative",
            "inference_allowed": False,
        }
    )
    temporal_policy = policy_envelope(
        {
            "policy_id": "historical_proxy_fixture_v1",
            "temporal_alignment_requirement": "input_provenance_proxy_allowed",
            "temporal_alignment_class": "controller_input_provenance_proxy",
            "physical_presentation_verified": False,
        }
    )
    members: list[dict[str, Any]] = [
        {
            "source_ordinal": 1,
            "dataset_id": "dataset-a",
            "recording_id": "recording-a",
            "analysis_zarr": str(analysis_root / "recording-a.zarr"),
            "protocol_names": ["behavior-a"],
            "protocol_hashes": [_digest("protocol-a")],
            "source_member_sha256": canonical_json_sha256(source_members[0]),
            "source_subject_ids": ["capture-subject-reused"],
            "source_subject_identity_status": "capture_time_non_authoritative",
            "acquisition_batch_id": None,
            "acquisition_batch_identity_status": "missing_historical_not_inferred",
            "analysis_unit_kind": "recording",
            "analysis_unit_id": "recording-a",
            "membership_state": "admitted",
            "reason_code": None,
            "disposition_evidence": {
                "evidence_type": "fixture_decision_v1",
                "detail": "exact authority accepted",
                "path": None,
                "file_sha256": None,
                "record_sha256": None,
            },
            "admission_receipts": [
                {
                    "role": "validated_behavior_receipt",
                    "path": str(admission_path.resolve()),
                    "file_sha256": _sha256_file(admission_path),
                    "record_sha256": _digest("admission-record"),
                    "schema_id": "fixture.validated_behavior_receipt",
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
            "disposition_evidence": {
                "evidence_type": "fixture_decision_v1",
                "detail": "source authority is invalid",
                "path": None,
                "file_sha256": None,
                "record_sha256": None,
            },
            "admission_receipts": [],
        },
    ]
    membership = build_validated_behavior_cohort_membership(
        membership_id="mixed-behavior-fixture-v1",
        source_membership={
            "adapter_id": "fixture_source_v1",
            "schema_id": "fixture.closed_membership",
            "schema_version": 1,
            "profile": "fixture_source_v1",
            "path": str(source_path.resolve()),
            "file_sha256": _sha256_file(source_path),
            "record_sha256": _digest("source-membership"),
            "member_count": 2,
            "source_members_sha256": canonical_json_sha256(
                [canonical_json_sha256(item) for item in source_members]
            ),
        },
        members=members,
        analysis_zarr_root=analysis_root,
        admission_receipt_root=receipt_root,
        analysis_unit_policy=unit_policy,
        acquisition_batch_policy=batch_policy,
        temporal_alignment_policy=temporal_policy,
        palette_commit=COMMIT,
        created_at_utc=NOW,
    )
    membership_path = tmp_path / "membership.json"
    _write_json(membership_path, membership)

    bundle_path = bundle_root / "recording-a.bundle.json"
    _write_json(bundle_path, {"bundle": "recording-a"})
    capability_contract = build_capability_contract(
        profile_id="generic_behavior_fixture_v1",
        keys=("semantic_epochs", "swim_bouts"),
        reason_codes_by_state={
            "complete": (None,),
            "inapplicable": ("not_requested",),
            "invalid": ("invalid_source",),
            "review_required": ("review_pending",),
            "stale": ("source_stale",),
            "unavailable": ("blocked_by_invalid_membership",),
        },
    )
    complete_capabilities = {
        key: {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": {"scope": "fixture", "key": key},
        }
        for key in capability_contract["keys"]
    }
    invalid_capabilities = {
        key: {
            "state": "unavailable",
            "reason_code": "blocked_by_invalid_membership",
            "detail": "parent member is invalid",
            "binding": None,
        }
        for key in capability_contract["keys"]
    }
    bundle_set = build_validated_behavior_bundle_set(
        bundle_set_id="mixed-behavior-bundles-v1",
        membership=membership,
        membership_path=membership_path,
        membership_file_sha256=_sha256_file(membership_path),
        bundle_root=bundle_root,
        bundle_profile={"adapter_id": "fixture_bundle_v1"},
        capability_contract=capability_contract,
        members=[
            {
                "recording_id": "recording-a",
                "bundle_state": "complete",
                "reason_code": None,
                "bundle": {
                    "adapter_id": "fixture_bundle_v1",
                    "path": str(bundle_path.resolve()),
                    "file_sha256": _sha256_file(bundle_path),
                    "record_sha256": _digest("bundle-record-a"),
                    "schema_id": "fixture.recording_bundle",
                    "schema_version": 1,
                    "method_id": "fixture_bundle_v1",
                    "status": "complete",
                    "receipt_bindings": [
                        {
                            "role": "validated_behavior_receipt",
                            "path": str(admission_path.resolve()),
                            "file_sha256": _sha256_file(admission_path),
                            "record_sha256": _digest("admission-record"),
                            "schema_id": "fixture.validated_behavior_receipt",
                            "schema_version": 1,
                        }
                    ],
                    "binding_inventory_sha256": _digest("binding-inventory-a"),
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
    bundle_set_path = tmp_path / "bundle-set.json"
    _write_json(bundle_set_path, bundle_set)
    return membership_path, bundle_set_path


def _plan(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    membership_path, bundle_set_path = _membership_and_bundle_set(tmp_path)
    plan = build_validated_behavior_export_plan(
        membership_path=membership_path,
        bundle_set_path=bundle_set_path,
        export_run_id="mixed_behavior_export_v1",
        shard_root=(tmp_path / "shards").resolve(),
        publication_root=(tmp_path / "publication").resolve(),
        palette_commit=COMMIT,
        palette_repo=Path(__file__).resolve().parents[3],
        created_at_utc=NOW,
    )
    plan_path = tmp_path / "export-plan.json"
    write_validated_behavior_export_plan(plan_path, plan)
    return plan_path, plan


def _complete_shards(plan_path: Path) -> list[dict[str, Any]]:
    return [
        write_validated_behavior_recording_shard(
            plan_path=plan_path,
            member_ordinal=ordinal,
            created_at_utc=NOW,
        )
        for ordinal in (1, 2)
    ]


def _legacy_plan(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    current_path, current = _plan(tmp_path)
    legacy_body = dict(current)
    legacy_body.pop("plan_sha256")
    legacy_body.pop("evidence_profile")
    legacy_body["schema_version"] = 1
    legacy_body["method_id"] = "closed_membership_recording_shard_plan_v1"
    legacy = {
        **legacy_body,
        "plan_sha256": canonical_json_sha256(legacy_body),
    }
    legacy_path = current_path.with_name("legacy-export-plan.json")
    _write_json(legacy_path, legacy)
    return legacy_path, legacy


def _publish_legacy_v1_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Build a tiny real-Parquet archive in the previously published grammar."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    plan_path, plan = _legacy_plan(tmp_path)
    plan, membership, bundle_set = cohort_export.read_validated_behavior_export_plan(
        plan_path
    )
    receipts: list[tuple[Path, dict[str, Any]]] = []
    table_names = tuple(sorted(CORE_TABLE_SPECS))
    for member, membership_member, bundle_member in zip(
        plan["members"], membership["members"], bundle_set["members"], strict=True
    ):
        shard_root = cohort_export.planned_shard_root(plan, member)
        parts: dict[str, dict[str, Any]] = {}
        zero_reasons: dict[str, None] = {}
        for table_name in table_names:
            spec = CORE_TABLE_SPECS[table_name]
            rows, zero_reason = cohort_export._core_rows(  # noqa: SLF001
                table_name, plan, membership_member, bundle_member
            )
            assert zero_reason is None
            table_dir = shard_root / "tables" / table_name
            table_dir.mkdir(parents=True, exist_ok=True)
            part = table_dir / "part-00000.parquet"
            schema = cohort_export.exact_schema(
                spec.contract,
                metadata=cohort_export._part_footer(  # noqa: SLF001
                    plan=plan, member=member, table_name=table_name
                ),
            )
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), part)
            semantics = cohort_export._resident_row_semantics(  # noqa: SLF001
                rows,
                spec,
                export_run_id=plan["export_run_id"],
                recording_id=member["recording_id"],
            )
            parts[table_name] = cohort_export._part_receipt(  # noqa: SLF001
                part=part,
                relative_path=f"tables/{table_name}/part-00000.parquet",
                row_count=len(rows),
                spec=spec,
                key_bounds=semantics["primary_key_bounds"],
            )
            zero_reasons[table_name] = None
        receipt_body = {
            "schema_id": cohort_export.SHARD_SCHEMA_ID,
            "schema_version": cohort_export.LEGACY_SHARD_SCHEMA_VERSION,
            "method_id": cohort_export.LEGACY_SHARD_METHOD_ID,
            "status": cohort_export.SHARD_STATUS,
            "export_run_id": plan["export_run_id"],
            "export_plan": {
                "path": str(plan_path),
                "file_sha256": _sha256_file(plan_path),
                "plan_sha256": plan["plan_sha256"],
            },
            "member": {
                "ordinal": member["ordinal"],
                "recording_id": member["recording_id"],
                "analysis_zarr": member["analysis_zarr"],
                "membership_member_sha256": member["membership_member_sha256"],
                "bundle_set_member_sha256": member["bundle_set_member_sha256"],
                "bundle_state": member["bundle_state"],
                "capabilities_sha256": member["capabilities_sha256"],
            },
            "membership": plan["membership"],
            "bundle_set": plan["bundle_set"],
            "requested_tables": list(table_names),
            "table_coverage": plan["table_coverage"],
            "parts_by_table": parts,
            "zero_row_reasons_by_table": zero_reasons,
            "parameters": plan["parameters"],
            "validation_policy": cohort_export.LEGACY_SHARD_VALIDATION_POLICY,
            "software_authority": plan["software_authority"],
            "created_at_utc": NOW,
            "safety": cohort_export.SAFETY,
        }
        receipt = cohort_export._sealed(  # noqa: SLF001
            receipt_body, digest_field="record_sha256"
        )
        receipt_path = shard_root / "receipt.json"
        _write_json(receipt_path, receipt)
        receipts.append((receipt_path, receipt))

    generation_id = "legacy-generation"
    generation_relative = cohort_export._generation_relative_path(  # noqa: SLF001
        plan["export_run_id"], generation_id
    )
    publication_root = Path(plan["publication_root"])
    generation_root = publication_root / generation_relative
    inventory: dict[str, list[dict[str, Any]]] = {name: [] for name in table_names}
    roster: list[dict[str, Any]] = []
    for member, (receipt_path, receipt) in zip(plan["members"], receipts, strict=True):
        receipt_inside = (
            Path("provenance")
            / "shard_receipts"
            / f"member={member['ordinal']:06d}-{member['recording_id']}.json"
        )
        published_receipt = generation_root / receipt_inside
        published_receipt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(receipt_path, published_receipt)
        roster.append(
            {
                "member_ordinal": member["ordinal"],
                "recording_id": member["recording_id"],
                "source_path": str(receipt_path),
                "path": (generation_relative / receipt_inside).as_posix(),
                "size_bytes": published_receipt.stat().st_size,
                "file_sha256": _sha256_file(published_receipt),
                "record_sha256": receipt["record_sha256"],
            }
        )
        for table_name in table_names:
            source_entry = receipt["parts_by_table"][table_name]
            part_inside = (
                Path("tables")
                / table_name
                / f"member={member['ordinal']:06d}-{member['recording_id']}"
                / "part-00000.parquet"
            )
            target = generation_root / part_inside
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(receipt_path.parent / source_entry["path"], target)
            inventory[table_name].append(
                {
                    **source_entry,
                    "member_ordinal": member["ordinal"],
                    "recording_id": member["recording_id"],
                    "path": (generation_relative / part_inside).as_posix(),
                    "generation_path": generation_relative.as_posix(),
                    "source_shard_record_sha256": receipt["record_sha256"],
                }
            )
    roster_files, semantic_sha = (
        cohort_export._validate_published_shard_roster(  # noqa: SLF001
            generation_root,
            generation_relative_path=generation_relative.as_posix(),
            plan=plan,
            roster=roster,
        )
    )
    assert semantic_sha is None
    validation = cohort_export._global_validate_generation(  # noqa: SLF001
        generation_root,
        generation_relative_path=generation_relative.as_posix(),
        plan=plan,
        inventory=inventory,
        table_specs=CORE_TABLE_SPECS,
        hash_parts=True,
        additional_expected_files=roster_files,
    )
    inventory_sha = canonical_json_sha256(inventory)
    roster_sha = canonical_json_sha256(roster)
    validation_body = {
        "schema_id": cohort_export.VALIDATION_RECEIPT_SCHEMA_ID,
        "schema_version": cohort_export.LEGACY_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "export_run_id": plan["export_run_id"],
        "export_plan_sha256": plan["plan_sha256"],
        "generation_id": generation_id,
        "generation_path": generation_relative.as_posix(),
        "part_inventory_sha256": inventory_sha,
        "shard_receipts_sha256": roster_sha,
        "validation_policy": cohort_export.LEGACY_VALIDATION_POLICY,
        "validation_result": validation,
        "software_authority": plan["software_authority"],
        "validated_at_utc": NOW,
        "safety": cohort_export.SAFETY,
    }
    validation_receipt = cohort_export._sealed(  # noqa: SLF001
        validation_body, digest_field="record_sha256"
    )
    validation_path = generation_root / "validation" / "receipt.json"
    _write_json(validation_path, validation_receipt)
    publication = {
        "schema_id": cohort_export.PUBLICATION_SCHEMA_ID,
        "schema_version": cohort_export.PUBLICATION_SCHEMA_VERSION,
        "state": "complete",
        "generation_id": generation_id,
        "generation_path": generation_relative.as_posix(),
        "parts_by_table": inventory,
        "part_inventory_sha256": inventory_sha,
    }
    manifest_body = {
        "schema_id": cohort_export.EXPORT_SCHEMA_ID,
        "schema_version": cohort_export.LEGACY_EXPORT_SCHEMA_VERSION,
        "method_id": cohort_export.LEGACY_EXPORT_METHOD_ID,
        "status": cohort_export.EXPORT_STATUS,
        "export_run_id": plan["export_run_id"],
        "export_plan": {
            "path": str(plan_path),
            "file_sha256": _sha256_file(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
        "export_profile": plan["export_profile"],
        "membership": plan["membership"],
        "bundle_set": plan["bundle_set"],
        "member_count": plan["member_count"],
        "membership_state_counts": membership["state_counts"],
        "bundle_state_counts": bundle_set["state_counts"],
        "capability_matrix_sha256": bundle_set["capability_matrix_sha256"],
        "table_names": list(table_names),
        "table_specs": plan["table_specs"],
        "table_coverage": plan["table_coverage"],
        "arrow_schema_contracts": plan["arrow_schema_contracts"],
        "shard_receipts": roster,
        "shard_receipts_sha256": roster_sha,
        "row_counts_by_table": validation["row_counts_by_table"],
        "parameters": plan["parameters"],
        "analysis_unit_policy": membership["analysis_unit_policy"],
        "acquisition_batch_policy": membership["acquisition_batch_policy"],
        "temporal_alignment_policy": membership["temporal_alignment_policy"],
        "publication": publication,
        "validation_receipt": {
            "path": (generation_relative / "validation" / "receipt.json").as_posix(),
            "size_bytes": validation_path.stat().st_size,
            "file_sha256": _sha256_file(validation_path),
            "record_sha256": validation_receipt["record_sha256"],
        },
        "software_authority": plan["software_authority"],
        "created_at_utc": NOW,
        "safety": cohort_export.SAFETY,
    }
    manifest = cohort_export._sealed(  # noqa: SLF001
        manifest_body, digest_field="record_sha256"
    )
    manifest_path = cohort_export.validated_behavior_manifest_path(
        publication_root, plan["export_run_id"]
    )
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def test_core_contracts_are_generic_and_do_not_masquerade_as_v3() -> None:
    assert CORE_TABLE_NAMES == (
        "cohort_recordings",
        "recording_bundles",
        "recording_capabilities",
    )
    assert all(
        spec.contract.schema_id.startswith(
            "palette.analytics.validated_behavior.table."
        )
        for spec in CORE_TABLE_SPECS.values()
    )
    assert all(
        "chaser" not in spec.contract.schema_id for spec in CORE_TABLE_SPECS.values()
    )


def test_recording_shards_are_deterministic_exact_and_reusable(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)

    first = write_validated_behavior_recording_shard(
        plan_path=plan_path, member_ordinal=1, created_at_utc=NOW
    )
    second = write_validated_behavior_recording_shard(
        plan_path=plan_path, member_ordinal=1, created_at_utc=NOW
    )

    assert first["reused"] is False
    assert second["reused"] is True
    assert second["record_sha256"] == first["record_sha256"]
    assert first["parts_by_table"]["cohort_recordings"]["row_count"] == 1
    assert first["parts_by_table"]["recording_capabilities"]["row_count"] == 2
    assert Path(first["receipt_path"]) == planned_shard_receipt_path(
        plan, plan["members"][0]
    )


def test_current_plan_declares_receipt_composed_evidence(tmp_path: Path) -> None:
    _plan_path, plan = _plan(tmp_path)

    assert plan["schema_version"] == 2
    assert plan["method_id"] == "closed_membership_recording_shard_plan_v2"
    assert plan["evidence_profile"] == cohort_export._current_evidence_profile()
    assert plan["evidence_profile"]["normal_finalization_payload_decoding"] is False
    assert plan["evidence_profile"]["required_shard_receipt"]["schema_version"] == 2


def test_legacy_plan_remains_readable_but_cannot_enter_current_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_path, legacy = _legacy_plan(tmp_path)

    observed, _membership, _bundle_set = (
        cohort_export.read_validated_behavior_export_plan(legacy_path)
    )
    assert observed == legacy
    monkeypatch.setattr(
        cohort_export,
        "_global_validate_generation",
        lambda *_args, **_kwargs: pytest.fail("legacy plan reached payload validation"),
    )
    with pytest.raises(
        ValidatedBehaviorExportError,
        match="legacy plans remain read-only",
    ):
        publish_validated_behavior_cohort(
            plan_path=legacy_path,
            generation_id="legacy-not-runnable",
            created_at_utc=NOW,
        )


def test_real_legacy_v1_publication_remains_readable(tmp_path: Path) -> None:
    _manifest_path, expected = _publish_legacy_v1_fixture(tmp_path)
    plan_path = Path(expected["export_plan"]["path"])
    legacy_plan, _membership, _bundle_set = (
        cohort_export.read_validated_behavior_export_plan(plan_path)
    )

    receipt_manifest, _membership, _bundle_set = (
        read_validated_behavior_export_manifest(
            legacy_plan["publication_root"],
            legacy_plan["export_run_id"],
            validate_parts="receipt",
        )
    )
    full_manifest, _membership, _bundle_set = read_validated_behavior_export_manifest(
        legacy_plan["publication_root"],
        legacy_plan["export_run_id"],
        validate_parts="full",
    )
    assert receipt_manifest == expected
    assert full_manifest == expected


def test_receipt_composed_finalizer_hashes_each_destination_once_without_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    real_sha256_file = cohort_export.sha256_file
    parquet_hashes: list[Path] = []

    def tracked_sha256(path: Path) -> str:
        resolved = Path(path)
        if resolved.suffix == ".parquet":
            parquet_hashes.append(resolved)
        return real_sha256_file(resolved)

    def decoded_scan_forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("normal finalization attempted a decoded relation scan")

    monkeypatch.setattr(cohort_export, "sha256_file", tracked_sha256)
    monkeypatch.setattr(
        cohort_export, "_observed_primary_key_summary", decoded_scan_forbidden
    )
    monkeypatch.setattr(cohort_export, "_part_relation_values", decoded_scan_forbidden)
    monkeypatch.setattr(
        cohort_export, "_part_foreign_key_is_closed", decoded_scan_forbidden
    )
    monkeypatch.setattr(
        cohort_export, "_foreign_key_observation", decoded_scan_forbidden
    )

    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="receipt-composed",
        created_at_utc=NOW,
    )

    assert len(parquet_hashes) == len(CORE_TABLE_SPECS) * plan["member_count"]
    assert len(set(parquet_hashes)) == len(parquet_hashes)
    assert all(".staging" in path.parts for path in parquet_hashes)
    assert published["schema_version"] == 2
    assert published["validation_receipt"]["record_sha256"]
    assert published["transfer_receipt"]["record_sha256"]
    assert published["process_telemetry"]["copied_part_count"] == len(parquet_hashes)
    persisted = json.loads(Path(published["manifest_path"]).read_text(encoding="utf-8"))
    assert "process_telemetry" not in persisted
    parquet_hashes.clear()
    reopened, _membership, _bundle_set = read_validated_behavior_export_manifest(
        plan["publication_root"], plan["export_run_id"]
    )
    assert reopened["record_sha256"] == published["record_sha256"]
    assert parquet_hashes == []


def test_invalid_v2_semantic_proof_blocks_without_payload_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    receipt_path = planned_shard_receipt_path(plan, plan["members"][0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    semantic = receipt["semantic_validation"]
    semantic["table_results"]["cohort_recordings"]["row_owner_validation"][
        "status"
    ] = "incomplete"
    semantic_body = dict(semantic)
    semantic_body.pop("record_sha256")
    semantic["record_sha256"] = canonical_json_sha256(semantic_body)
    receipt_body = dict(receipt)
    receipt_body.pop("record_sha256")
    receipt["record_sha256"] = canonical_json_sha256(receipt_body)
    receipt_path.chmod(0o644)
    _write_json(receipt_path, receipt)
    receipt_path.chmod(0o444)

    monkeypatch.setattr(
        cohort_export,
        "_global_validate_generation",
        lambda *_args, **_kwargs: pytest.fail("invalid proof reached payload fallback"),
    )
    with pytest.raises(
        ValidatedBehaviorExportError,
        match="semantic table proof is invalid or incomplete",
    ):
        publish_validated_behavior_cohort(
            plan_path=plan_path,
            generation_id="invalid-semantic-proof",
            created_at_utc=NOW,
        )


def test_copy_mutation_fails_before_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    real_copyfile = cohort_export.shutil.copyfile
    mutated = False

    def mutating_copy(source: Path, destination: Path) -> str:
        nonlocal mutated
        result = real_copyfile(source, destination)
        target = Path(destination)
        if not mutated and target.suffix == ".parquet":
            with target.open("ab") as stream:
                stream.write(b"copy-mutation")
            mutated = True
        return str(result)

    monkeypatch.setattr(cohort_export.shutil, "copyfile", mutating_copy)
    with pytest.raises(
        ValidatedBehaviorExportError,
        match="destination bytes differ",
    ):
        publish_validated_behavior_cohort(
            plan_path=plan_path,
            generation_id="copy-mutation",
            created_at_utc=NOW,
        )
    assert mutated is True
    assert not Path(
        cohort_export.validated_behavior_manifest_path(
            plan["publication_root"], plan["export_run_id"]
        )
    ).exists()


def test_changed_source_part_blocks_before_copy_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    receipt_path = planned_shard_receipt_path(plan, plan["members"][0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_part = (
        receipt_path.parent / receipt["parts_by_table"]["cohort_recordings"]["path"]
    )
    source_part.chmod(0o644)
    with source_part.open("ab") as stream:
        stream.write(b"source-mutation")
    source_part.chmod(0o444)

    monkeypatch.setattr(
        cohort_export.shutil,
        "copyfile",
        lambda *_args, **_kwargs: pytest.fail("stale source reached copy"),
    )
    with pytest.raises(
        ValidatedBehaviorExportError,
        match="part size differs from its receipt",
    ):
        publish_validated_behavior_cohort(
            plan_path=plan_path,
            generation_id="stale-source",
            created_at_utc=NOW,
        )


def test_same_size_source_mutation_is_caught_by_destination_digest(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    receipt_path = planned_shard_receipt_path(plan, plan["members"][0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_part = (
        receipt_path.parent / receipt["parts_by_table"]["cohort_recordings"]["path"]
    )
    payload = bytearray(source_part.read_bytes())
    assert len(payload) > 16
    payload[8] ^= 1
    source_part.chmod(0o644)
    source_part.write_bytes(payload)
    source_part.chmod(0o444)

    with pytest.raises(
        ValidatedBehaviorExportError,
        match="destination bytes differ",
    ):
        publish_validated_behavior_cohort(
            plan_path=plan_path,
            generation_id="same-size-stale-source",
            created_at_utc=NOW,
        )


def test_scientific_adapter_is_capability_gated_without_protocol_logic_in_core(
    tmp_path: Path,
) -> None:
    membership_path, bundle_set_path = _membership_and_bundle_set(tmp_path)
    event_contract = ArrowTableContract(
        table_name="fixture_behavior_events",
        schema_namespace="fixture.analytics.validated_behavior.table",
        fields=(
            field("export_run_id", "string"),
            field("recording_id", "string"),
            field("event_id", "int64"),
            field("value", "float64"),
        ),
        primary_key=("export_run_id", "recording_id", "event_id"),
    )
    event_spec = ValidatedBehaviorTableSpec(
        contract=event_contract,
        grain="one row per exact fixture event",
        capability_policy="optional_explicit_coverage",
        required_capability="semantic_epochs",
        zero_rows_allowed=True,
    )
    specs = {**CORE_TABLE_SPECS, event_spec.table_name: event_spec}
    plan = build_validated_behavior_export_plan(
        membership_path=membership_path,
        bundle_set_path=bundle_set_path,
        export_run_id="capability_gated_export_v1",
        shard_root=(tmp_path / "gated-shards").resolve(),
        publication_root=(tmp_path / "gated-publication").resolve(),
        palette_commit=COMMIT,
        palette_repo=Path(__file__).resolve().parents[3],
        table_specs=specs,
        export_profile_id="fixture_behavior_extension_v1",
        created_at_utc=NOW,
    )
    plan_path = tmp_path / "gated-plan.json"
    write_validated_behavior_export_plan(plan_path, plan)
    called_recordings: list[str] = []

    def extract(
        plan_value: dict[str, Any],
        membership_member: dict[str, Any],
        _bundle_member: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], None]:
        called_recordings.append(membership_member["recording_id"])
        return [
            {
                "export_run_id": plan_value["export_run_id"],
                "recording_id": membership_member["recording_id"],
                "event_id": 1,
                "value": 2.5,
            }
        ], None

    first = write_validated_behavior_recording_shard(
        plan_path=plan_path,
        member_ordinal=1,
        table_specs=specs,
        row_extractors={event_spec.table_name: extract},
        created_at_utc=NOW,
    )
    second = write_validated_behavior_recording_shard(
        plan_path=plan_path,
        member_ordinal=2,
        table_specs=specs,
        row_extractors={event_spec.table_name: extract},
        created_at_utc=NOW,
    )

    assert called_recordings == ["recording-a"]
    assert first["parts_by_table"][event_spec.table_name]["row_count"] == 1
    assert second["parts_by_table"][event_spec.table_name]["row_count"] == 0
    assert second["zero_row_reasons_by_table"][event_spec.table_name] == (
        "capability-unavailable-blocked_by_invalid_membership"
    )
    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        table_specs=specs,
        generation_id="generation-gated",
        created_at_utc=NOW,
    )
    dataset = ValidatedBehaviorExportDataset.open(
        plan["publication_root"],
        plan["export_run_id"],
        table_specs=specs,
    )
    assert published["row_counts_by_table"][event_spec.table_name] == 1
    assert dataset.table(event_spec.table_name).collect_bounded(
        max_rows=5
    ).to_dicts() == [
        {
            "export_run_id": "capability_gated_export_v1",
            "recording_id": "recording-a",
            "event_id": 1,
            "value": 2.5,
        }
    ]


def test_dense_column_batches_write_with_constant_memory_key_validation(
    tmp_path: Path,
) -> None:
    membership_path, bundle_set_path = _membership_and_bundle_set(tmp_path)
    contract = ArrowTableContract(
        table_name="fixture_dense_samples",
        schema_namespace="fixture.analytics.validated_behavior.table",
        fields=(
            field("export_run_id", "string"),
            field("recording_id", "string"),
            field("sample_id", "int64"),
            field("value", "float32"),
        ),
        primary_key=("export_run_id", "recording_id", "sample_id"),
    )
    spec = ValidatedBehaviorTableSpec(
        contract=contract,
        grain="one row per exact dense fixture sample",
        capability_policy="optional_explicit_coverage",
        required_capability="semantic_epochs",
        zero_rows_allowed=True,
        primary_key_validation="strictly_increasing_v1",
    )
    specs = {**CORE_TABLE_SPECS, spec.table_name: spec}
    plan = build_validated_behavior_export_plan(
        membership_path=membership_path,
        bundle_set_path=bundle_set_path,
        export_run_id="dense_column_batch_export_v1",
        shard_root=(tmp_path / "dense-shards").resolve(),
        publication_root=(tmp_path / "dense-publication").resolve(),
        palette_commit=COMMIT,
        palette_repo=Path(__file__).resolve().parents[3],
        table_specs=specs,
        export_profile_id="dense_column_batch_profile_v1",
        created_at_utc=NOW,
    )
    plan_path = tmp_path / "dense-plan.json"
    write_validated_behavior_export_plan(plan_path, plan)

    def extract(
        plan_value: dict[str, Any],
        member: dict[str, Any],
        _bundle_member: dict[str, Any],
    ) -> ValidatedBehaviorBatchSource:
        common = {
            "export_run_id": [plan_value["export_run_id"]],
            "recording_id": [member["recording_id"]],
        }
        return ValidatedBehaviorBatchSource(
            batches=(
                {
                    **common,
                    "sample_id": [0],
                    "value": [1.0],
                },
                {
                    "export_run_id": [plan_value["export_run_id"]] * 2,
                    "recording_id": [member["recording_id"]] * 2,
                    "sample_id": [1, 2],
                    "value": [2.0, 3.0],
                },
            )
        )

    receipt = write_validated_behavior_recording_shard(
        plan_path=plan_path,
        member_ordinal=1,
        table_specs=specs,
        row_extractors={spec.table_name: extract},
        created_at_utc=NOW,
    )

    part = receipt["parts_by_table"][spec.table_name]
    assert part["row_count"] == 3
    assert part["primary_key_bounds"] == {
        "minimum": ["dense_column_batch_export_v1", "recording-a", 0],
        "maximum": ["dense_column_batch_export_v1", "recording-a", 2],
    }


def test_dense_column_batches_reject_nonincreasing_primary_keys(
    tmp_path: Path,
) -> None:
    membership_path, bundle_set_path = _membership_and_bundle_set(tmp_path)
    contract = ArrowTableContract(
        table_name="fixture_dense_order_failure",
        schema_namespace="fixture.analytics.validated_behavior.table",
        fields=(
            field("export_run_id", "string"),
            field("recording_id", "string"),
            field("sample_id", "int64"),
        ),
        primary_key=("export_run_id", "recording_id", "sample_id"),
    )
    spec = ValidatedBehaviorTableSpec(
        contract=contract,
        grain="one row per exact dense fixture sample",
        capability_policy="optional_explicit_coverage",
        required_capability="semantic_epochs",
        zero_rows_allowed=True,
        primary_key_validation="strictly_increasing_v1",
    )
    specs = {**CORE_TABLE_SPECS, spec.table_name: spec}
    plan = build_validated_behavior_export_plan(
        membership_path=membership_path,
        bundle_set_path=bundle_set_path,
        export_run_id="dense_order_failure_v1",
        shard_root=(tmp_path / "dense-order-shards").resolve(),
        publication_root=(tmp_path / "dense-order-publication").resolve(),
        palette_commit=COMMIT,
        palette_repo=Path(__file__).resolve().parents[3],
        table_specs=specs,
        export_profile_id="dense_order_failure_profile_v1",
        created_at_utc=NOW,
    )
    plan_path = tmp_path / "dense-order-plan.json"
    write_validated_behavior_export_plan(plan_path, plan)

    def extract(
        plan_value: dict[str, Any],
        member: dict[str, Any],
        _bundle_member: dict[str, Any],
    ) -> ValidatedBehaviorBatchSource:
        return ValidatedBehaviorBatchSource(
            batches=(
                {
                    "export_run_id": [plan_value["export_run_id"]] * 2,
                    "recording_id": [member["recording_id"]] * 2,
                    "sample_id": [1, 1],
                },
            )
        )

    with pytest.raises(
        ValidatedBehaviorExportError,
        match="primary keys are not strictly increasing",
    ):
        write_validated_behavior_recording_shard(
            plan_path=plan_path,
            member_ordinal=1,
            table_specs=specs,
            row_extractors={spec.table_name: extract},
            created_at_utc=NOW,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "inexact field set"),
        ("extra", "inexact field set"),
        ("null", "null required fields"),
    ),
)
def test_scientific_adapter_rows_fail_closed_before_arrow_normalization(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    membership_path, bundle_set_path = _membership_and_bundle_set(tmp_path)
    event_contract = ArrowTableContract(
        table_name="strict_fixture_events",
        schema_namespace="fixture.analytics.validated_behavior.table",
        fields=(
            field("export_run_id", "string"),
            field("recording_id", "string"),
            field("event_id", "int64"),
            field("value", "float64"),
        ),
        primary_key=("export_run_id", "recording_id", "event_id"),
    )
    event_spec = ValidatedBehaviorTableSpec(
        contract=event_contract,
        grain="one row per strict fixture event",
        capability_policy="optional_explicit_coverage",
        required_capability="semantic_epochs",
        zero_rows_allowed=True,
    )
    specs = {**CORE_TABLE_SPECS, event_spec.table_name: event_spec}
    plan = build_validated_behavior_export_plan(
        membership_path=membership_path,
        bundle_set_path=bundle_set_path,
        export_run_id=f"strict_row_{mutation}_v1",
        shard_root=(tmp_path / "strict-shards").resolve(),
        publication_root=(tmp_path / "strict-publication").resolve(),
        palette_commit=COMMIT,
        palette_repo=Path(__file__).resolve().parents[3],
        table_specs=specs,
        export_profile_id="strict_fixture_extension_v1",
        created_at_utc=NOW,
    )
    plan_path = tmp_path / "strict-plan.json"
    write_validated_behavior_export_plan(plan_path, plan)

    def extract(
        plan_value: dict[str, Any],
        membership_member: dict[str, Any],
        _bundle_member: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], None]:
        row: dict[str, Any] = {
            "export_run_id": plan_value["export_run_id"],
            "recording_id": membership_member["recording_id"],
            "event_id": 1,
            "value": 2.5,
        }
        if mutation == "missing":
            del row["value"]
        elif mutation == "extra":
            row["undeclared"] = "unsafe"
        else:
            row["value"] = None
        return [row], None

    with pytest.raises(ValidatedBehaviorExportError, match=message):
        write_validated_behavior_recording_shard(
            plan_path=plan_path,
            member_ordinal=1,
            table_specs=specs,
            row_extractors={event_spec.table_name: extract},
            created_at_utc=NOW,
        )


def test_foreign_key_validation_reads_non_primary_local_columns(
    tmp_path: Path,
) -> None:
    membership_path, bundle_set_path = _membership_and_bundle_set(tmp_path)
    entity_contract = ArrowTableContract(
        table_name="fixture_entities",
        schema_namespace="fixture.analytics.validated_behavior.table",
        fields=(
            field("export_run_id", "string"),
            field("recording_id", "string"),
            field("entity_id", "int64"),
        ),
        primary_key=("export_run_id", "recording_id", "entity_id"),
    )
    event_contract = ArrowTableContract(
        table_name="fixture_entity_events",
        schema_namespace="fixture.analytics.validated_behavior.table",
        fields=(
            field("export_run_id", "string"),
            field("recording_id", "string"),
            field("event_id", "int64"),
            field("entity_id", "int64"),
        ),
        primary_key=("export_run_id", "recording_id", "event_id"),
    )
    recording_fk = (
        (
            ("export_run_id", "recording_id"),
            "cohort_recordings",
            ("export_run_id", "recording_id"),
        ),
    )
    specs = {
        **CORE_TABLE_SPECS,
        "fixture_entities": ValidatedBehaviorTableSpec(
            contract=entity_contract,
            grain="one row per fixture entity",
            capability_policy="optional_explicit_coverage",
            required_capability="semantic_epochs",
            foreign_keys=recording_fk,
            zero_rows_allowed=True,
        ),
        "fixture_entity_events": ValidatedBehaviorTableSpec(
            contract=event_contract,
            grain="one row per fixture event",
            capability_policy="optional_explicit_coverage",
            required_capability="semantic_epochs",
            foreign_keys=recording_fk
            + (
                (
                    ("export_run_id", "recording_id", "entity_id"),
                    "fixture_entities",
                    ("export_run_id", "recording_id", "entity_id"),
                ),
            ),
            zero_rows_allowed=True,
        ),
    }

    def run_export(*, run_id: str, child_entity_id: int) -> dict[str, Any]:
        plan = build_validated_behavior_export_plan(
            membership_path=membership_path,
            bundle_set_path=bundle_set_path,
            export_run_id=run_id,
            shard_root=(tmp_path / f"{run_id}-shards").resolve(),
            publication_root=(tmp_path / f"{run_id}-publication").resolve(),
            palette_commit=COMMIT,
            palette_repo=Path(__file__).resolve().parents[3],
            table_specs=specs,
            export_profile_id="non_primary_foreign_key_fixture_v1",
            created_at_utc=NOW,
        )
        plan_path = tmp_path / f"{run_id}-plan.json"
        write_validated_behavior_export_plan(plan_path, plan)

        def entity_rows(
            plan_value: dict[str, Any],
            member: dict[str, Any],
            _bundle_member: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], None]:
            return [
                {
                    "export_run_id": plan_value["export_run_id"],
                    "recording_id": member["recording_id"],
                    "entity_id": 7,
                }
            ], None

        def event_rows(
            plan_value: dict[str, Any],
            member: dict[str, Any],
            _bundle_member: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], None]:
            return [
                {
                    "export_run_id": plan_value["export_run_id"],
                    "recording_id": member["recording_id"],
                    "event_id": 1,
                    "entity_id": child_entity_id,
                }
            ], None

        extractors = {
            "fixture_entities": entity_rows,
            "fixture_entity_events": event_rows,
        }
        for ordinal in (1, 2):
            write_validated_behavior_recording_shard(
                plan_path=plan_path,
                member_ordinal=ordinal,
                table_specs=specs,
                row_extractors=extractors,
                created_at_utc=NOW,
            )
        return publish_validated_behavior_cohort(
            plan_path=plan_path,
            table_specs=specs,
            generation_id=f"generation-{run_id}",
            created_at_utc=NOW,
        )

    published = run_export(run_id="non_primary_fk_valid_v1", child_entity_id=7)
    assert published["row_counts_by_table"]["fixture_entity_events"] == 1

    with pytest.raises(
        ValidatedBehaviorExportError,
        match="fixture_entity_events: foreign key to fixture_entities is not closed",
    ):
        run_export(run_id="non_primary_fk_invalid_v1", child_entity_id=8)


def test_plan_rejects_scientific_capability_absent_from_bundle_profile(
    tmp_path: Path,
) -> None:
    membership_path, bundle_set_path = _membership_and_bundle_set(tmp_path)
    missing_contract = ArrowTableContract(
        table_name="missing_authority_events",
        schema_namespace="fixture.analytics.validated_behavior.table",
        fields=(
            field("export_run_id", "string"),
            field("recording_id", "string"),
            field("event_id", "int64"),
        ),
        primary_key=("export_run_id", "recording_id", "event_id"),
    )
    specs = {
        **CORE_TABLE_SPECS,
        "missing_authority_events": ValidatedBehaviorTableSpec(
            contract=missing_contract,
            grain="one row per unavailable authority event",
            capability_policy="optional_explicit_coverage",
            required_capability="not_in_bundle_contract",
            zero_rows_allowed=True,
        ),
    }

    with pytest.raises(
        ValidatedBehaviorExportError, match="absent from the bundle profile"
    ):
        build_validated_behavior_export_plan(
            membership_path=membership_path,
            bundle_set_path=bundle_set_path,
            export_run_id="missing_capability_export_v1",
            shard_root=(tmp_path / "missing-shards").resolve(),
            publication_root=(tmp_path / "missing-publication").resolve(),
            palette_commit=COMMIT,
            palette_repo=Path(__file__).resolve().parents[3],
            table_specs=specs,
            export_profile_id="missing_capability_profile_v1",
            created_at_utc=NOW,
        )


def test_missing_shard_blocks_publication_without_manifest(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    write_validated_behavior_recording_shard(
        plan_path=plan_path, member_ordinal=1, created_at_utc=NOW
    )

    with pytest.raises(FileNotFoundError):
        publish_validated_behavior_cohort(
            plan_path=plan_path,
            generation_id="generation-a",
            created_at_utc=NOW,
        )

    manifest = (
        Path(plan["publication_root"])
        / "validated_behavior"
        / "v1"
        / "manifests"
        / "export_run_id=mixed_behavior_export_v1.json"
    )
    assert not manifest.exists()


def test_publication_and_lazy_reader_preserve_closed_states(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="generation-a",
        created_at_utc=NOW,
    )

    assert published["status"] == "complete_selector_ineligible"
    assert published["safety"]["selector_eligible"] is False
    assert published["row_counts_by_table"] == {
        "cohort_recordings": 2,
        "recording_bundles": 2,
        "recording_capabilities": 4,
    }
    assert all(
        len(entries) == 2
        for entries in published["publication"]["parts_by_table"].values()
    )

    dataset = ValidatedBehaviorExportDataset.open(
        plan["publication_root"], plan["export_run_id"]
    )
    assert dataset.validation_mode == "receipt"
    assert dataset.cache_identity == published["record_sha256"]
    invalid = dataset.table("cohort_recordings").collect_bounded(
        max_rows=10,
        columns=("recording_id", "membership_state", "reason_code"),
        predicate=pl.col("membership_state") == "invalid",
    )
    assert invalid.to_dicts() == [
        {
            "recording_id": "recording-b",
            "membership_state": "invalid",
            "reason_code": "invalid_source_authority",
        }
    ]
    capabilities = dataset.table("recording_capabilities").collect_bounded(
        max_rows=3,
        columns=("recording_id", "capability_id", "state"),
    )
    assert capabilities.height == 3
    assert (
        dataset.table("recording_capabilities").query_identity()[
            "export_manifest_record_sha256"
        ]
        == published["record_sha256"]
    )
    full_dataset = ValidatedBehaviorExportDataset.open(
        plan["publication_root"],
        plan["export_run_id"],
        full_part_hashes=True,
    )
    assert full_dataset.validation_mode == "full"
    assert full_dataset.cache_identity == dataset.cache_identity


def test_reader_rejects_manifest_unselected_file_in_generation(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="generation-a",
        created_at_utc=NOW,
    )
    generation = (
        Path(plan["publication_root"]) / published["publication"]["generation_path"]
    )
    (generation / "tables" / "cohort_recordings" / "unselected.parquet").write_bytes(
        b"not selected"
    )

    with pytest.raises(
        ValidatedBehaviorExportError, match="outside its closed inventory"
    ):
        ValidatedBehaviorExportDataset.open(
            plan["publication_root"], plan["export_run_id"]
        )


def test_receipt_reader_rejects_unfrozen_v2_part(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="generation-unfrozen",
        created_at_utc=NOW,
    )
    first_part = (
        Path(plan["publication_root"])
        / next(iter(published["publication"]["parts_by_table"].values()))[0]["path"]
    )
    first_part.chmod(0o644)

    with pytest.raises(
        ValidatedBehaviorExportError,
        match="not cooperatively frozen read-only",
    ):
        read_validated_behavior_export_manifest(
            plan["publication_root"], plan["export_run_id"]
        )


def test_reader_rejects_resealed_transfer_receipt_not_bound_by_manifest(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="generation-transfer-tamper",
        created_at_utc=NOW,
    )
    transfer_path = (
        Path(plan["publication_root"]) / published["transfer_receipt"]["path"]
    )
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    transfer["staging_attempt_id"] = "another-staging-attempt"
    transfer_body = dict(transfer)
    transfer_body.pop("record_sha256")
    transfer["record_sha256"] = canonical_json_sha256(transfer_body)
    transfer_path.chmod(0o644)
    _write_json(transfer_path, transfer)
    transfer_path.chmod(0o444)

    with pytest.raises(
        ValidatedBehaviorExportError,
        match="Transfer-receipt file binding is stale",
    ):
        read_validated_behavior_export_manifest(
            plan["publication_root"], plan["export_run_id"]
        )


def test_reader_rejects_manifest_body_mutation(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="generation-manifest-tamper",
        created_at_utc=NOW,
    )
    manifest_path = Path(published["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "tampered"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValidatedBehaviorExportError, match="self digest is stale"):
        read_validated_behavior_export_manifest(
            plan["publication_root"], plan["export_run_id"]
        )


def test_reader_rejects_symbolic_link_manifest_alias(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    published = publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="generation-a",
        created_at_utc=NOW,
    )
    manifest_path = Path(published["manifest_path"])
    real_manifest = manifest_path.with_name("stored-manifest.json")
    manifest_path.rename(real_manifest)
    manifest_path.symlink_to(real_manifest)

    with pytest.raises(FileNotFoundError, match="symbolic-link alias"):
        ValidatedBehaviorExportDataset.open(
            plan["publication_root"], plan["export_run_id"]
        )


def test_changed_membership_bytes_block_shard_reuse_and_reader(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    _complete_shards(plan_path)
    publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="generation-a",
        created_at_utc=NOW,
    )
    membership_path = Path(plan["membership"]["path"])
    membership_path.write_text(
        membership_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(
        ValidatedBehaviorExportError, match="absent, aliased, or changed"
    ):
        write_validated_behavior_recording_shard(
            plan_path=plan_path, member_ordinal=1, created_at_utc=NOW
        )
    with pytest.raises(
        ValidatedBehaviorExportError, match="absent, aliased, or changed"
    ):
        read_validated_behavior_export_manifest(
            plan["publication_root"], plan["export_run_id"]
        )


def test_publication_is_immutable_for_same_run_id(tmp_path: Path) -> None:
    plan_path, _plan_value = _plan(tmp_path)
    _complete_shards(plan_path)
    publish_validated_behavior_cohort(
        plan_path=plan_path,
        generation_id="generation-a",
        created_at_utc=NOW,
    )

    with pytest.raises(FileExistsError):
        publish_validated_behavior_cohort(
            plan_path=plan_path,
            generation_id="generation-b",
            created_at_utc=NOW,
        )


def test_cli_plan_emits_bounded_core_profile_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    membership_path, bundle_set_path = _membership_and_bundle_set(tmp_path)
    repository = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(
        export_cli,
        "_require_clean_current_authority",
        lambda _expected=None: (repository, COMMIT),
    )
    plan_path = tmp_path / "cli-plan.json"

    assert (
        export_cli.main(
            [
                "plan",
                "--membership",
                str(membership_path),
                "--bundle-set",
                str(bundle_set_path),
                "--export-run-id",
                "cli_core_export_v1",
                "--plan-output",
                str(plan_path),
                "--shard-root",
                str((tmp_path / "cli-shards").resolve()),
                "--publication-root",
                str((tmp_path / "cli-publication").resolve()),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["profile_id"] == "validated_behavior_core_metadata_v1"
    assert summary["member_count"] == 2
    assert summary["table_names"] == list(CORE_TABLE_NAMES)
    assert summary["safety"]["selector_eligible"] is False
    assert plan_path.is_file()


def test_lsf_submitter_renders_bounded_array_and_success_barrier(
    tmp_path: Path,
) -> None:
    plan_path, _plan_value = _plan(tmp_path)
    repository = Path(__file__).resolve().parents[3]
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        'case " $* " in\n'
        f"  *\" rev-parse HEAD \"*) printf '{COMMIT}\\n';;\n"
        '  *" status --porcelain "*) :;;\n'
        "  *) exit 2;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    log_dir = tmp_path / "lsf-logs"

    result = subprocess.run(
        [
            "bash",
            str(
                repository
                / "scripts"
                / "submit_validated_behavior_cohort_export_bsub.sh"
            ),
            "--plan",
            str(plan_path),
            "--palette-repo",
            str(repository),
            "--log-dir",
            str(log_dir),
            "--max-active",
            "3",
            "--queue",
            "short",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "mode=render-only" in result.stdout
    assert "member_count=2" in result.stdout
    assert r"vb_mixed_behavior_export_v1\[1-2\]%3" in result.stdout
    assert "finalize_bsub_command=" in result.stdout
    assert "shard-array-job-id" in result.stdout
    run_dir = log_dir / "validated_behavior_export_mixed_behavior_export_v1"
    shard_job = (run_dir / "run_shard.sh").read_text(encoding="utf-8")
    finalizer = (run_dir / "run_finalize.sh").read_text(encoding="utf-8")
    assert '--member-ordinal "${LSB_JOBINDEX}"' in shard_job
    assert "materialize_validated_behavior_cohort_export finalize" in finalizer
    assert "materialize_validated_behavior_cohort_export validate" in finalizer
    assert "registry" not in shard_job.casefold()


def test_lsf_submitter_rejects_legacy_plan_before_creating_scratch(
    tmp_path: Path,
) -> None:
    legacy_plan_path, _legacy_plan_value = _legacy_plan(tmp_path)
    repository = Path(__file__).resolve().parents[3]
    log_dir = tmp_path / "legacy-lsf-logs"

    result = subprocess.run(
        [
            "bash",
            str(
                repository
                / "scripts"
                / "submit_validated_behavior_cohort_export_bsub.sh"
            ),
            "--plan",
            str(legacy_plan_path),
            "--palette-repo",
            str(repository),
            "--log-dir",
            str(log_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "legacy plans remain read-only" in result.stderr
    assert not log_dir.exists()
