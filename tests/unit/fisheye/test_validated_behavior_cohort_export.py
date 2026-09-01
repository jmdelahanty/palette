from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
