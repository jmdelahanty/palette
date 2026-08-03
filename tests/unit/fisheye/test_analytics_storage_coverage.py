from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from fisheye.analysis_workflows.analytics_storage_coverage import (
    ANALYTICS_STORAGE_COVERAGE_SCHEMA_ID,
    ANALYTICS_STORAGE_COVERAGE_SCHEMA_VERSION,
    AnalyticsStorageCoverageError,
    analytics_storage_coverage_json,
    build_analytics_storage_coverage_report,
    validate_analytics_storage_coverage_report,
)
from fisheye.analysis_workflows.storage_candidate_catalog import (
    DERIVED_ANALYSIS_STORAGE_CANDIDATES,
)
from fisheye.analysis_workflows.storage_contract_catalog import (
    DERIVED_ANALYSIS_STORAGE_CONTRACTS,
)
from fisheye.analysis_workflows.surface_classification_catalog import (
    ANALYTICS_SURFACE_CLASSIFICATIONS,
)
from fisheye.registry.stage_catalog import DERIVED_ANALYSIS, STAGE_SPECS, StageSpec


def test_live_report_covers_every_maintained_derived_stage_exactly_once() -> None:
    report = build_analytics_storage_coverage_report()

    expected_stage_ids = sorted(
        spec.id
        for spec in STAGE_SPECS
        if spec.category == DERIVED_ANALYSIS and not spec.deprecated
    )
    coverage = report["derived_stage_coverage"]
    assert [record["stage_id"] for record in coverage] == expected_stage_ids
    assert report["schema_id"] == ANALYTICS_STORAGE_COVERAGE_SCHEMA_ID
    assert report["schema_version"] == ANALYTICS_STORAGE_COVERAGE_SCHEMA_VERSION

    owner_by_stage = {
        record["stage_id"]: (record["owner_kind"], record["owner_id"])
        for record in coverage
    }
    assert owner_by_stage["track_kinematics_visualization"] == (
        "classified_non_catalog_surface",
        "track_kinematics_visualization",
    )
    assert all(
        kind == "central_storage_contract"
        for stage_id, (kind, _owner_id) in owner_by_stage.items()
        if stage_id != "track_kinematics_visualization"
    )


def test_report_preserves_live_storage_and_classification_statuses() -> None:
    report = build_analytics_storage_coverage_report()
    contracts = {record["stage_id"]: record for record in report["storage_contracts"]}
    candidates = {
        record["stage_id"]: record for record in report["storage_candidates"]
    }
    surfaces = {
        record["surface_id"]: record for record in report["classified_surfaces"]
    }

    assert len(contracts) == len(DERIVED_ANALYSIS_STORAGE_CONTRACTS) == 9
    assert len(candidates) == len(DERIVED_ANALYSIS_STORAGE_CANDIDATES) == 7
    assert len(surfaces) == len(ANALYTICS_SURFACE_CLASSIFICATIONS) == 22
    assert report["summary"] == {
        "derived_stage_count": 10,
        "central_storage_contract_count": 9,
        "storage_candidate_count": 7,
        "atomic_storage_candidate_count": 5,
        "guarded_direct_storage_candidate_count": 2,
        "classified_non_catalog_stage_count": 1,
        "classified_surface_count": 22,
        "additional_classified_surface_count": 21,
        "byte_planner_adopted_count": 0,
        "serialized_registry_publication_count": 9,
        "exact_contract_required_surface_count": 18,
        "central_catalog_adoption_pending_surface_count": 4,
    }
    assert contracts["eye_angles"]["byte_planner_adopted"] is False
    assert contracts["eye_angles"]["registry_publication"] == (
        "serialized_finalizer_v1"
    )
    assert candidates["eye_angles"]["profile_id"] == (
        "eye_angle_access_aware_candidate_v1"
    )
    assert all(
        record["selector_eligible"] is False
        and record["profile_promoted"] is False
        for record in candidates.values()
    )
    assert surfaces["chaser_distance"]["central_storage_catalog_required"] is True
    assert (
        surfaces["track_kinematics_visualization"]["exact_storage_contract_status"]
        == "not_applicable"
    )


def test_report_contains_every_additional_classified_surface() -> None:
    report = build_analytics_storage_coverage_report()

    assert {record["surface_id"] for record in report["classified_surfaces"]} == {
        surface.surface_id for surface in ANALYTICS_SURFACE_CLASSIFICATIONS
    }


def test_new_maintained_derived_stage_fails_closed_when_unclassified() -> None:
    added_stage = StageSpec(
        id="new_analytics_surface",
        artifact_families=("analysis/new_analytics_surface_runs",),
        category=DERIVED_ANALYSIS,
    )

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match="maintained derived stage 'new_analytics_surface' is unclassified",
    ):
        build_analytics_storage_coverage_report(stage_specs=(*STAGE_SPECS, added_stage))


def test_duplicate_central_storage_ownership_fails_closed() -> None:
    duplicate = DERIVED_ANALYSIS_STORAGE_CONTRACTS[0]

    with pytest.raises(
        AnalyticsStorageCoverageError, match="duplicate central storage ownership"
    ):
        build_analytics_storage_coverage_report(
            storage_contracts=(*DERIVED_ANALYSIS_STORAGE_CONTRACTS, duplicate)
        )


def test_duplicate_classified_surface_id_fails_closed() -> None:
    duplicate = ANALYTICS_SURFACE_CLASSIFICATIONS[0]

    with pytest.raises(
        AnalyticsStorageCoverageError, match="duplicate classified surface ids"
    ):
        build_analytics_storage_coverage_report(
            surface_classifications=(*ANALYTICS_SURFACE_CLASSIFICATIONS, duplicate)
        )


def test_duplicate_storage_candidate_ownership_fails_closed() -> None:
    duplicate = DERIVED_ANALYSIS_STORAGE_CANDIDATES[0]

    with pytest.raises(
        AnalyticsStorageCoverageError, match="duplicate storage candidate ownership"
    ):
        build_analytics_storage_coverage_report(
            storage_candidates=(*DERIVED_ANALYSIS_STORAGE_CANDIDATES, duplicate)
        )


def test_central_and_classified_ownership_of_one_derived_stage_fails_closed() -> None:
    existing = next(
        surface
        for surface in ANALYTICS_SURFACE_CLASSIFICATIONS
        if surface.surface_id == "track_kinematics_visualization"
    )
    conflicting = replace(
        existing,
        surface_id="eye_angle_visualization_conflict",
        stage_binding="eye_angles",
    )

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match="duplicate central and classified ownership",
    ):
        build_analytics_storage_coverage_report(
            surface_classifications=(*ANALYTICS_SURFACE_CLASSIFICATIONS, conflicting)
        )


def test_live_stage_artifact_family_must_match_its_declared_owner() -> None:
    eye_stage = next(stage for stage in STAGE_SPECS if stage.id == "eye_angles")
    changed = replace(
        eye_stage,
        artifact_families=("analysis/not_eye_angle_runs",),
    )
    stages = tuple(changed if stage is eye_stage else stage for stage in STAGE_SPECS)

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match="artifact families do not match its declared owner",
    ):
        build_analytics_storage_coverage_report(stage_specs=stages)


def test_classified_fallback_cannot_hide_required_central_adoption() -> None:
    existing = next(
        surface
        for surface in ANALYTICS_SURFACE_CLASSIFICATIONS
        if surface.surface_id == "stimulus_epochs"
    )
    required_fallback = replace(
        existing,
        surface_id="track_kinematics_visualization",
        stage_binding="track_kinematics_visualization",
    )
    surfaces = tuple(
        required_fallback
        if surface.surface_id == "track_kinematics_visualization"
        else surface
        for surface in ANALYTICS_SURFACE_CLASSIFICATIONS
    )

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match="requires the central catalog but has only classified ownership",
    ):
        build_analytics_storage_coverage_report(surface_classifications=surfaces)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda report: report.__setitem__("schema_version", True),
            "schema_version is unsupported",
        ),
        (
            lambda report: report.__setitem__("unexpected", "field"),
            "report has wrong fields",
        ),
        (
            lambda report: report["summary"].__setitem__("derived_stage_count", True),
            "derived_stage_count must be an exact nonnegative int",
        ),
        (
            lambda report: report["storage_contracts"][0].__setitem__(
                "byte_planner_adopted", 0
            ),
            "byte_planner_adopted must be an exact bool",
        ),
        (
            lambda report: report["classified_surfaces"][0].__setitem__(
                "schema_version", False
            ),
            "schema_version must be an exact positive int",
        ),
        (
            lambda report: report["storage_candidates"][0].__setitem__(
                "selector_eligible", 0
            ),
            "selector_eligible must be an exact bool",
        ),
    ),
)
def test_report_validator_rejects_malformed_exact_fields_and_types(
    mutate: object, message: str
) -> None:
    report = deepcopy(build_analytics_storage_coverage_report())
    mutate(report)

    with pytest.raises(AnalyticsStorageCoverageError, match=message):
        validate_analytics_storage_coverage_report(report)


def test_report_validator_rejects_serialized_duplicate_ownership() -> None:
    report = deepcopy(build_analytics_storage_coverage_report())
    duplicate = deepcopy(report["derived_stage_coverage"][0])
    report["derived_stage_coverage"].append(duplicate)
    report["summary"]["derived_stage_count"] += 1

    with pytest.raises(AnalyticsStorageCoverageError, match="duplicate derived stage"):
        validate_analytics_storage_coverage_report(report)


def test_report_validator_rejects_serialized_owner_path_drift() -> None:
    report = deepcopy(build_analytics_storage_coverage_report())
    report["derived_stage_coverage"][0]["artifact_families"] = [
        "analysis/not_the_declared_owner"
    ]

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match="artifact_families do not match its declared owner",
    ):
        validate_analytics_storage_coverage_report(report)


@pytest.mark.parametrize("field", ("selector_eligible", "profile_promoted"))
def test_report_validator_rejects_candidate_promotion_claim(field: str) -> None:
    report = deepcopy(build_analytics_storage_coverage_report())
    report["storage_candidates"][0][field] = True

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match=rf"{field} must remain false",
    ):
        validate_analytics_storage_coverage_report(report)


def test_report_validator_rejects_candidate_publication_semantic_drift() -> None:
    report = deepcopy(build_analytics_storage_coverage_report())
    atomic = next(
        record
        for record in report["storage_candidates"]
        if record["publication_mode"] == "shared_atomic_nonpromoting_v1"
    )
    atomic["repairs_failed_visibility"] = False

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match="violates storage-candidate semantics",
    ):
        validate_analytics_storage_coverage_report(report)


def test_report_validator_rejects_candidate_without_logical_contract() -> None:
    report = deepcopy(build_analytics_storage_coverage_report())
    candidate = report["storage_candidates"][-1]
    report["storage_contracts"] = [
        contract
        for contract in report["storage_contracts"]
        if contract["stage_id"] != candidate["stage_id"]
    ]
    report["summary"]["central_storage_contract_count"] -= 1

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match="storage candidates contain stages absent from logical contracts",
    ):
        validate_analytics_storage_coverage_report(report)


def test_report_validator_rejects_cross_field_surface_tampering() -> None:
    report = deepcopy(build_analytics_storage_coverage_report())
    gate = next(
        record
        for record in report["classified_surfaces"]
        if record["surface_id"] == "registered_detection_gate"
    )
    gate["lifecycle"] = "legacy"

    with pytest.raises(
        AnalyticsStorageCoverageError,
        match="violates classified-surface semantics",
    ):
        validate_analytics_storage_coverage_report(report)


def test_canonical_json_is_stable_strict_and_round_trips() -> None:
    first = analytics_storage_coverage_json()
    second = analytics_storage_coverage_json(
        deepcopy(build_analytics_storage_coverage_report())
    )

    assert first == second
    assert first.endswith("\n")
    assert "NaN" not in first
    decoded = json.loads(first)
    validate_analytics_storage_coverage_report(decoded)
    assert analytics_storage_coverage_json(decoded) == first
