"""Machine-readable coverage of maintained analytics storage surfaces.

The report is derived from the live stage, storage-contract, and surface
classification catalogs.  It deliberately does not infer promotion: planner,
publication, and exact-contract states are copied verbatim from their owning
catalogs.  Building the report is also the CI drift guard: every maintained
``DERIVED_ANALYSIS`` stage must have exactly one owner, either a central storage
contract or one explicitly classified non-catalog surface.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import NoReturn

from fisheye.analysis_workflows.storage_contract_catalog import (
    DERIVED_ANALYSIS_STORAGE_CONTRACTS,
    DerivedAnalysisStorageContract,
)
from fisheye.analysis_workflows.storage_candidate_catalog import (
    DERIVED_ANALYSIS_STORAGE_CANDIDATES,
    DerivedAnalysisStorageCandidate,
    StorageCandidatePublicationMode,
)
from fisheye.analysis_workflows.surface_classification_catalog import (
    ANALYTICS_SURFACE_CLASSIFICATIONS,
    AnalyticsConsumerScope,
    AnalyticsLifecycleContext,
    AnalyticsMutationMode,
    AnalyticsStorageKind,
    AnalyticsSurfaceClass,
    AnalyticsSurfaceClassification,
    AnalyticsSurfaceLifecycle,
    ExactStorageContractStatus,
)
from fisheye.registry.stage_catalog import DERIVED_ANALYSIS, STAGE_SPECS, StageSpec


ANALYTICS_STORAGE_COVERAGE_SCHEMA_ID = "palette.analytics.storage_coverage"
ANALYTICS_STORAGE_COVERAGE_SCHEMA_VERSION = 3

_REPORT_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "derived_stage_coverage",
        "storage_contracts",
        "storage_candidates",
        "classified_surfaces",
        "summary",
    }
)
_STORAGE_CANDIDATE_FIELDS = frozenset(
    {
        "stage_id",
        "run_parent",
        "profile_id",
        "owner_module",
        "entrypoint",
        "publication_mode",
        "consolidates_before_return",
        "repairs_failed_visibility",
        "selector_eligible",
        "profile_promoted",
    }
)
_STAGE_COVERAGE_FIELDS = frozenset(
    {"stage_id", "artifact_families", "owner_kind", "owner_id"}
)
_STORAGE_CONTRACT_FIELDS = frozenset(
    {
        "stage_id",
        "run_parent",
        "availability_parents",
        "schema_id",
        "schema_version",
        "method",
        "method_version",
        "layout",
        "materializer_module",
        "publication_owner_module",
        "physical_policy_owner",
        "registry_publication",
        "byte_planner_adopted",
        "publication_owner_kind",
        "publication_entrypoint",
    }
)
_CLASSIFIED_SURFACE_FIELDS = frozenset(
    {
        "surface_id",
        "classification",
        "lifecycle",
        "lifecycle_context",
        "consumer_scope",
        "stage_binding",
        "owner_module",
        "owner_entrypoint",
        "owner_path",
        "storage_kind",
        "mutation_mode",
        "array_bearing",
        "exact_storage_contract_status",
        "central_storage_catalog_required",
        "schema_id",
        "schema_version",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "derived_stage_count",
        "central_storage_contract_count",
        "storage_candidate_count",
        "atomic_storage_candidate_count",
        "guarded_direct_storage_candidate_count",
        "classified_non_catalog_stage_count",
        "classified_surface_count",
        "additional_classified_surface_count",
        "byte_planner_adopted_count",
        "serialized_registry_publication_count",
        "exact_contract_required_surface_count",
        "central_catalog_adoption_pending_surface_count",
    }
)


class AnalyticsStorageCoverageError(ValueError):
    """The live catalogs or a serialized coverage report are inconsistent."""


def _fail(message: str) -> NoReturn:
    raise AnalyticsStorageCoverageError(message)


def _exact_str(value: object, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        _fail(f"{field} must be an exact {'possibly empty ' if allow_empty else ''}string")
    return value


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        _fail(f"{field} must be an exact bool")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{field} must be an exact positive int")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{field} must be an exact nonnegative int")
    return value


def _exact_fields(value: object, expected: frozenset[str], *, field: str) -> Mapping[str, object]:
    if type(value) is not dict:
        _fail(f"{field} must be an exact object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        _fail(f"{field} has wrong fields: missing={missing}, unexpected={unexpected}")
    return value


def _exact_list(value: object, *, field: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{field} must be an exact array")
    return value


def _optional_exact_str(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _exact_str(value, field=field)


def _string_or_int(value: object, *, field: str) -> str | int:
    if type(value) is str and value:
        return value
    if type(value) is int:
        return value
    _fail(f"{field} must be an exact nonempty string or exact int")


def _validate_source_catalogs(
    *,
    stage_specs: Sequence[StageSpec],
    storage_contracts: Sequence[DerivedAnalysisStorageContract],
    surface_classifications: Sequence[AnalyticsSurfaceClassification],
) -> tuple[
    tuple[StageSpec, ...],
    tuple[DerivedAnalysisStorageContract, ...],
    tuple[AnalyticsSurfaceClassification, ...],
]:
    stages = tuple(stage_specs)
    contracts = tuple(storage_contracts)
    surfaces = tuple(surface_classifications)

    for index, stage in enumerate(stages):
        if type(stage) is not StageSpec:
            _fail(f"stage_specs[{index}] must be an exact StageSpec")
        _exact_str(stage.id, field=f"stage_specs[{index}].id")
        _exact_str(stage.category, field=f"stage_specs[{index}].category")
        _exact_bool(stage.deprecated, field=f"stage_specs[{index}].deprecated")
        if type(stage.artifact_families) is not tuple or any(
            type(value) is not str or not value for value in stage.artifact_families
        ):
            _fail(f"stage_specs[{index}].artifact_families must be exact strings")
    for index, contract in enumerate(contracts):
        if type(contract) is not DerivedAnalysisStorageContract:
            _fail(
                f"storage_contracts[{index}] must be an exact "
                "DerivedAnalysisStorageContract"
            )
    for index, surface in enumerate(surfaces):
        if type(surface) is not AnalyticsSurfaceClassification:
            _fail(
                f"surface_classifications[{index}] must be an exact "
                "AnalyticsSurfaceClassification"
            )

    derived = tuple(
        stage
        for stage in stages
        if stage.category == DERIVED_ANALYSIS and not stage.deprecated
    )
    derived_ids = [stage.id for stage in derived]
    duplicate_derived = sorted(
        stage_id for stage_id in set(derived_ids) if derived_ids.count(stage_id) > 1
    )
    if duplicate_derived:
        _fail(f"duplicate maintained derived stage ids: {duplicate_derived}")

    contract_ids = [contract.stage_id for contract in contracts]
    duplicate_contracts = sorted(
        stage_id for stage_id in set(contract_ids) if contract_ids.count(stage_id) > 1
    )
    if duplicate_contracts:
        _fail(f"duplicate central storage ownership: {duplicate_contracts}")
    orphan_contracts = sorted(set(contract_ids) - set(derived_ids))
    if orphan_contracts:
        _fail(
            "central storage contracts do not bind maintained derived stages: "
            f"{orphan_contracts}"
        )

    surface_ids = [surface.surface_id for surface in surfaces]
    duplicate_surfaces = sorted(
        surface_id for surface_id in set(surface_ids) if surface_ids.count(surface_id) > 1
    )
    if duplicate_surfaces:
        _fail(f"duplicate classified surface ids: {duplicate_surfaces}")

    contract_by_stage = {contract.stage_id: contract for contract in contracts}
    for stage in derived:
        classified = [
            surface for surface in surfaces if surface.stage_binding == stage.id
        ]
        central = contract_by_stage.get(stage.id)
        if central is not None and classified:
            _fail(
                f"derived stage {stage.id!r} has duplicate central and classified "
                "ownership"
            )
        if central is None and not classified:
            _fail(f"maintained derived stage {stage.id!r} is unclassified")
        if len(classified) > 1:
            _fail(
                f"derived stage {stage.id!r} has multiple classified owners: "
                f"{sorted(surface.surface_id for surface in classified)}"
            )
        if classified:
            owner = classified[0]
            if owner.central_storage_catalog_required:
                _fail(
                    f"derived stage {stage.id!r} requires the central catalog but "
                    "has only classified ownership"
                )
            if owner.lifecycle is not AnalyticsSurfaceLifecycle.CURRENT:
                _fail(
                    f"maintained derived stage {stage.id!r} has a non-current "
                    "classified owner"
                )
            expected_families = (owner.owner_path,)
        else:
            assert central is not None
            expected_families = (central.run_parent,)
        if stage.artifact_families != expected_families:
            _fail(
                f"maintained derived stage {stage.id!r} artifact families do not "
                f"match its declared owner: expected={expected_families}, "
                f"actual={stage.artifact_families}"
            )

    return derived, contracts, surfaces


def build_analytics_storage_coverage_report(
    *,
    stage_specs: Sequence[StageSpec] | None = None,
    storage_contracts: Sequence[DerivedAnalysisStorageContract] | None = None,
    storage_candidates: Sequence[DerivedAnalysisStorageCandidate] | None = None,
    surface_classifications: Sequence[AnalyticsSurfaceClassification] | None = None,
) -> dict[str, object]:
    """Build and validate the deterministic live analytics coverage report."""

    derived, contracts, surfaces = _validate_source_catalogs(
        stage_specs=STAGE_SPECS if stage_specs is None else stage_specs,
        storage_contracts=(
            DERIVED_ANALYSIS_STORAGE_CONTRACTS
            if storage_contracts is None
            else storage_contracts
        ),
        surface_classifications=(
            ANALYTICS_SURFACE_CLASSIFICATIONS
            if surface_classifications is None
            else surface_classifications
        ),
    )
    contract_by_stage = {contract.stage_id: contract for contract in contracts}
    candidates = tuple(
        DERIVED_ANALYSIS_STORAGE_CANDIDATES
        if storage_candidates is None
        else storage_candidates
    )
    for index, candidate in enumerate(candidates):
        if type(candidate) is not DerivedAnalysisStorageCandidate:
            _fail(
                f"storage_candidates[{index}] must be an exact "
                "DerivedAnalysisStorageCandidate"
            )
        if candidate.stage_id not in contract_by_stage:
            _fail(
                f"storage candidate {candidate.stage_id!r} has no report logical contract"
            )
    candidate_ids = [candidate.stage_id for candidate in candidates]
    duplicate_candidates = sorted(
        stage_id
        for stage_id in set(candidate_ids)
        if candidate_ids.count(stage_id) > 1
    )
    if duplicate_candidates:
        _fail(f"duplicate storage candidate ownership: {duplicate_candidates}")
    surface_by_stage = {
        surface.stage_binding: surface
        for surface in surfaces
        if surface.stage_binding in {stage.id for stage in derived}
    }

    storage_records = sorted(
        (contract.resolved_schema() for contract in contracts),
        key=lambda record: str(record["stage_id"]),
    )
    candidate_records = sorted(
        (candidate.as_record() for candidate in candidates),
        key=lambda record: str(record["stage_id"]),
    )
    surface_records = sorted(
        (surface.as_record() for surface in surfaces),
        key=lambda record: str(record["surface_id"]),
    )
    derived_records: list[dict[str, object]] = []
    for stage in sorted(derived, key=lambda item: item.id):
        contract = contract_by_stage.get(stage.id)
        surface = surface_by_stage.get(stage.id)
        if contract is not None:
            owner_kind = "central_storage_contract"
            owner_id = contract.stage_id
        else:
            assert surface is not None
            owner_kind = "classified_non_catalog_surface"
            owner_id = surface.surface_id
        derived_records.append(
            {
                "stage_id": stage.id,
                "artifact_families": list(stage.artifact_families),
                "owner_kind": owner_kind,
                "owner_id": owner_id,
            }
        )

    derived_ids = {record["stage_id"] for record in derived_records}
    additional_count = sum(
        record["stage_binding"] not in derived_ids for record in surface_records
    )
    report: dict[str, object] = {
        "schema_id": ANALYTICS_STORAGE_COVERAGE_SCHEMA_ID,
        "schema_version": ANALYTICS_STORAGE_COVERAGE_SCHEMA_VERSION,
        "derived_stage_coverage": derived_records,
        "storage_contracts": storage_records,
        "storage_candidates": candidate_records,
        "classified_surfaces": surface_records,
        "summary": {
            "derived_stage_count": len(derived_records),
            "central_storage_contract_count": len(storage_records),
            "storage_candidate_count": len(candidate_records),
            "atomic_storage_candidate_count": sum(
                record["publication_mode"]
                == StorageCandidatePublicationMode.SHARED_ATOMIC.value
                for record in candidate_records
            ),
            "guarded_direct_storage_candidate_count": sum(
                record["publication_mode"]
                == StorageCandidatePublicationMode.GUARDED_DIRECT.value
                for record in candidate_records
            ),
            "classified_non_catalog_stage_count": sum(
                record["owner_kind"] == "classified_non_catalog_surface"
                for record in derived_records
            ),
            "classified_surface_count": len(surface_records),
            "additional_classified_surface_count": additional_count,
            "byte_planner_adopted_count": sum(
                record["byte_planner_adopted"] is True
                for record in storage_records
            ),
            "serialized_registry_publication_count": sum(
                record["registry_publication"] == "serialized_finalizer_v1"
                for record in storage_records
            ),
            "exact_contract_required_surface_count": sum(
                record["exact_storage_contract_status"] == "required"
                for record in surface_records
            ),
            "central_catalog_adoption_pending_surface_count": sum(
                record["central_storage_catalog_required"] is True
                for record in surface_records
            ),
        },
    }
    validate_analytics_storage_coverage_report(report)
    return report


def _validate_storage_contract_record(value: object, *, index: int) -> None:
    field = f"storage_contracts[{index}]"
    record = _exact_fields(value, _STORAGE_CONTRACT_FIELDS, field=field)
    for name in (
        "stage_id",
        "run_parent",
        "schema_id",
        "physical_policy_owner",
        "registry_publication",
        "publication_owner_kind",
        "publication_owner_module",
    ):
        _exact_str(record[name], field=f"{field}.{name}")
    parents = _exact_list(record["availability_parents"], field=f"{field}.availability_parents")
    for parent_index, parent in enumerate(parents):
        _exact_str(parent, field=f"{field}.availability_parents[{parent_index}]")
    _positive_int(record["schema_version"], field=f"{field}.schema_version")
    _exact_bool(record["byte_planner_adopted"], field=f"{field}.byte_planner_adopted")
    _optional_exact_str(record["method"], field=f"{field}.method")
    _string_or_int(record["method_version"], field=f"{field}.method_version")
    _optional_exact_str(record["layout"], field=f"{field}.layout")
    _optional_exact_str(
        record["materializer_module"], field=f"{field}.materializer_module"
    )
    _optional_exact_str(
        record["publication_entrypoint"], field=f"{field}.publication_entrypoint"
    )
    if record["registry_publication"] not in {
        "not_implemented",
        "serialized_finalizer_v1",
    }:
        _fail(f"{field}.registry_publication is unsupported")
    if record["publication_owner_kind"] not in {
        "shared_atomic_materializer_v1",
        "guarded_direct_writer_v1",
    }:
        _fail(f"{field}.publication_owner_kind is unsupported")


def _validate_classified_surface_record(value: object, *, index: int) -> None:
    field = f"classified_surfaces[{index}]"
    record = _exact_fields(value, _CLASSIFIED_SURFACE_FIELDS, field=field)
    for name in (
        "surface_id",
        "classification",
        "lifecycle",
        "lifecycle_context",
        "consumer_scope",
        "stage_binding",
        "owner_module",
        "owner_entrypoint",
        "owner_path",
        "storage_kind",
        "mutation_mode",
        "exact_storage_contract_status",
    ):
        _exact_str(record[name], field=f"{field}.{name}")
    for name in ("array_bearing", "central_storage_catalog_required"):
        _exact_bool(record[name], field=f"{field}.{name}")
    if record["classification"] not in {item.value for item in AnalyticsSurfaceClass}:
        _fail(f"{field}.classification is unsupported")
    if record["lifecycle"] not in {item.value for item in AnalyticsSurfaceLifecycle}:
        _fail(f"{field}.lifecycle is unsupported")
    if record["lifecycle_context"] not in {
        item.value for item in AnalyticsLifecycleContext
    }:
        _fail(f"{field}.lifecycle_context is unsupported")
    if record["consumer_scope"] not in {item.value for item in AnalyticsConsumerScope}:
        _fail(f"{field}.consumer_scope is unsupported")
    if record["storage_kind"] not in {item.value for item in AnalyticsStorageKind}:
        _fail(f"{field}.storage_kind is unsupported")
    if record["mutation_mode"] not in {item.value for item in AnalyticsMutationMode}:
        _fail(f"{field}.mutation_mode is unsupported")
    if record["exact_storage_contract_status"] not in {
        item.value for item in ExactStorageContractStatus
    }:
        _fail(f"{field}.exact_storage_contract_status is unsupported")
    schema_id = _optional_exact_str(record["schema_id"], field=f"{field}.schema_id")
    schema_version = record["schema_version"]
    if (schema_id is None) != (schema_version is None):
        _fail(f"{field} schema identity must be declared together")
    if schema_version is not None:
        _positive_int(schema_version, field=f"{field}.schema_version")
    try:
        AnalyticsSurfaceClassification(
            surface_id=record["surface_id"],
            classification=AnalyticsSurfaceClass(record["classification"]),
            lifecycle=AnalyticsSurfaceLifecycle(record["lifecycle"]),
            lifecycle_context=AnalyticsLifecycleContext(
                record["lifecycle_context"]
            ),
            consumer_scope=AnalyticsConsumerScope(record["consumer_scope"]),
            stage_binding=record["stage_binding"],
            owner_module=record["owner_module"],
            owner_entrypoint=record["owner_entrypoint"],
            owner_path=record["owner_path"],
            storage_kind=AnalyticsStorageKind(record["storage_kind"]),
            mutation_mode=AnalyticsMutationMode(record["mutation_mode"]),
            array_bearing=record["array_bearing"],
            exact_storage_contract_status=ExactStorageContractStatus(
                record["exact_storage_contract_status"]
            ),
            central_storage_catalog_required=record[
                "central_storage_catalog_required"
            ],
            schema_id=schema_id,
            schema_version=schema_version,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{field} violates classified-surface semantics: {exc}")


def _validate_storage_candidate_record(value: object, *, index: int) -> None:
    field = f"storage_candidates[{index}]"
    record = _exact_fields(value, _STORAGE_CANDIDATE_FIELDS, field=field)
    for name in (
        "stage_id",
        "run_parent",
        "profile_id",
        "owner_module",
        "entrypoint",
        "publication_mode",
    ):
        _exact_str(record[name], field=f"{field}.{name}")
    for name in (
        "consolidates_before_return",
        "repairs_failed_visibility",
        "selector_eligible",
        "profile_promoted",
    ):
        _exact_bool(record[name], field=f"{field}.{name}")
    if record["selector_eligible"] is not False:
        _fail(f"{field}.selector_eligible must remain false")
    if record["profile_promoted"] is not False:
        _fail(f"{field}.profile_promoted must remain false")
    try:
        DerivedAnalysisStorageCandidate(
            stage_id=record["stage_id"],
            run_parent=record["run_parent"],
            profile_id=record["profile_id"],
            owner_module=record["owner_module"],
            entrypoint_attr=record["entrypoint"],
            publication_mode=StorageCandidatePublicationMode(
                record["publication_mode"]
            ),
            consolidates_before_return=record["consolidates_before_return"],
            repairs_failed_visibility=record["repairs_failed_visibility"],
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{field} violates storage-candidate semantics: {exc}")


def validate_analytics_storage_coverage_report(report: object) -> None:
    """Fail closed on malformed, duplicated, or internally inconsistent reports."""

    root = _exact_fields(report, _REPORT_FIELDS, field="report")
    if root["schema_id"] != ANALYTICS_STORAGE_COVERAGE_SCHEMA_ID:
        _fail("report.schema_id is unsupported")
    if root["schema_version"] != ANALYTICS_STORAGE_COVERAGE_SCHEMA_VERSION or type(
        root["schema_version"]
    ) is not int:
        _fail("report.schema_version is unsupported")

    stages = _exact_list(root["derived_stage_coverage"], field="derived_stage_coverage")
    contracts = _exact_list(root["storage_contracts"], field="storage_contracts")
    candidates = _exact_list(root["storage_candidates"], field="storage_candidates")
    surfaces = _exact_list(root["classified_surfaces"], field="classified_surfaces")
    summary = _exact_fields(root["summary"], _SUMMARY_FIELDS, field="summary")

    for index, value in enumerate(contracts):
        _validate_storage_contract_record(value, index=index)
    for index, value in enumerate(candidates):
        _validate_storage_candidate_record(value, index=index)
    for index, value in enumerate(surfaces):
        _validate_classified_surface_record(value, index=index)

    def reject_duplicates(values: list[str], *, field: str) -> None:
        duplicates = sorted(value for value in set(values) if values.count(value) > 1)
        if duplicates:
            _fail(f"duplicate {field}: {duplicates}")

    contract_ids = [item["stage_id"] for item in contracts]
    candidate_ids = [item["stage_id"] for item in candidates]
    surface_ids = [item["surface_id"] for item in surfaces]
    reject_duplicates(contract_ids, field="storage contract stage ids")
    reject_duplicates(candidate_ids, field="storage candidate stage ids")
    reject_duplicates(surface_ids, field="classified surface ids")
    if set(candidate_ids) - set(contract_ids):
        _fail("storage candidates contain stages absent from logical contracts")

    stage_ids: list[str] = []
    for index, value in enumerate(stages):
        field = f"derived_stage_coverage[{index}]"
        record = _exact_fields(value, _STAGE_COVERAGE_FIELDS, field=field)
        stage_id = _exact_str(record["stage_id"], field=f"{field}.stage_id")
        owner_id = _exact_str(record["owner_id"], field=f"{field}.owner_id")
        owner_kind = _exact_str(record["owner_kind"], field=f"{field}.owner_kind")
        artifact_families = _exact_list(
            record["artifact_families"], field=f"{field}.artifact_families"
        )
        for family_index, family in enumerate(artifact_families):
            _exact_str(
                family, field=f"{field}.artifact_families[{family_index}]"
            )
        if owner_kind not in {
            "central_storage_contract",
            "classified_non_catalog_surface",
        }:
            _fail(f"{field}.owner_kind is unsupported")
        stage_ids.append(stage_id)
        contract_matches = [
            item for item in contracts if item["stage_id"] == stage_id
        ]
        surface_matches = [
            item for item in surfaces if item["stage_binding"] == stage_id
        ]
        if owner_kind == "central_storage_contract":
            if owner_id != stage_id or len(contract_matches) != 1 or surface_matches:
                _fail(f"{field} does not have exactly one central owner")
            expected_families = [contract_matches[0]["run_parent"]]
        elif (
            len(surface_matches) != 1
            or owner_id != surface_matches[0]["surface_id"]
            or contract_matches
            or surface_matches[0]["central_storage_catalog_required"] is not False
            or surface_matches[0]["lifecycle"] != AnalyticsSurfaceLifecycle.CURRENT.value
        ):
            _fail(f"{field} does not have exactly one classified non-catalog owner")
        else:
            expected_families = [surface_matches[0]["owner_path"]]
        if artifact_families != expected_families:
            _fail(
                f"{field}.artifact_families do not match its declared owner"
            )

    reject_duplicates(stage_ids, field="derived stage ids")
    if set(contract_ids) - set(stage_ids):
        _fail("storage contracts contain stages absent from derived coverage")

    if stage_ids != sorted(stage_ids):
        _fail("derived_stage_coverage must be sorted by stage_id")
    if contract_ids != sorted(contract_ids):
        _fail("storage_contracts must be sorted by stage_id")
    if candidate_ids != sorted(candidate_ids):
        _fail("storage_candidates must be sorted by stage_id")
    if surface_ids != sorted(surface_ids):
        _fail("classified_surfaces must be sorted by surface_id")

    expected_summary = {
        "derived_stage_count": len(stages),
        "central_storage_contract_count": len(contracts),
        "storage_candidate_count": len(candidates),
        "atomic_storage_candidate_count": sum(
            item["publication_mode"]
            == StorageCandidatePublicationMode.SHARED_ATOMIC.value
            for item in candidates
        ),
        "guarded_direct_storage_candidate_count": sum(
            item["publication_mode"]
            == StorageCandidatePublicationMode.GUARDED_DIRECT.value
            for item in candidates
        ),
        "classified_non_catalog_stage_count": sum(
            item["owner_kind"] == "classified_non_catalog_surface" for item in stages
        ),
        "classified_surface_count": len(surfaces),
        "additional_classified_surface_count": sum(
            item["stage_binding"] not in set(stage_ids) for item in surfaces
        ),
        "byte_planner_adopted_count": sum(
            item["byte_planner_adopted"] is True for item in contracts
        ),
        "serialized_registry_publication_count": sum(
            item["registry_publication"] == "serialized_finalizer_v1"
            for item in contracts
        ),
        "exact_contract_required_surface_count": sum(
            item["exact_storage_contract_status"] == "required" for item in surfaces
        ),
        "central_catalog_adoption_pending_surface_count": sum(
            item["central_storage_catalog_required"] is True for item in surfaces
        ),
    }
    for name, expected in expected_summary.items():
        actual = _nonnegative_int(summary[name], field=f"summary.{name}")
        if actual != expected:
            _fail(f"summary.{name} must be {expected}, got {actual}")

    try:
        json.dumps(root, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AnalyticsStorageCoverageError(
            "report must be strict JSON-compatible"
        ) from exc


def analytics_storage_coverage_json(report: object | None = None) -> str:
    """Return one canonical strict-JSON representation with a trailing newline."""

    payload = build_analytics_storage_coverage_report() if report is None else report
    validate_analytics_storage_coverage_report(payload)
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> int:
    """Emit the live report; catalog drift fails before any JSON is printed."""

    print(analytics_storage_coverage_json(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYTICS_STORAGE_COVERAGE_SCHEMA_ID",
    "ANALYTICS_STORAGE_COVERAGE_SCHEMA_VERSION",
    "AnalyticsStorageCoverageError",
    "analytics_storage_coverage_json",
    "build_analytics_storage_coverage_report",
    "validate_analytics_storage_coverage_report",
]
