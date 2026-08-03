from dataclasses import replace
import importlib
import inspect
import json

import pytest

from fisheye.analysis.chaser_profiles import (
    full_chaser_analysis_profile_path,
    load_chaser_analysis_profile,
)
from fisheye.analysis_workflows.storage_contract_catalog import (
    DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE,
)
from fisheye.analysis_workflows.surface_classification_catalog import (
    ANALYTICS_SURFACE_CLASSIFICATION_BY_ID,
    ANALYTICS_SURFACE_CLASSIFICATIONS,
    CHASER_PROFILE_SURFACE_IDS,
    AnalyticsConsumerScope,
    AnalyticsLifecycleContext,
    AnalyticsMutationMode,
    AnalyticsSurfaceClass,
    AnalyticsSurfaceLifecycle,
    ExactStorageContractStatus,
    classified_surface_records,
)
from fisheye.registry.stage_catalog import get_stage_spec
from fisheye.shared.stage_run_groups import stage_run_parent_paths


def test_catalog_has_unique_closed_machine_readable_records() -> None:
    assert len(ANALYTICS_SURFACE_CLASSIFICATIONS) == len(
        ANALYTICS_SURFACE_CLASSIFICATION_BY_ID
    )
    assert len(ANALYTICS_SURFACE_CLASSIFICATIONS) == 22
    assert json.loads(json.dumps(classified_surface_records()))
    assert {entry.classification for entry in ANALYTICS_SURFACE_CLASSIFICATIONS} == set(
        AnalyticsSurfaceClass
    )
    assert {
        entry.exact_storage_contract_status
        for entry in ANALYTICS_SURFACE_CLASSIFICATIONS
    } == set(ExactStorageContractStatus)
    assert {entry.lifecycle for entry in ANALYTICS_SURFACE_CLASSIFICATIONS} == set(
        AnalyticsSurfaceLifecycle
    )
    assert {
        entry.lifecycle_context for entry in ANALYTICS_SURFACE_CLASSIFICATIONS
    } == set(AnalyticsLifecycleContext)
    assert {entry.consumer_scope for entry in ANALYTICS_SURFACE_CLASSIFICATIONS} == set(
        AnalyticsConsumerScope
    )


def test_every_classification_resolves_its_current_code_owner() -> None:
    for entry in ANALYTICS_SURFACE_CLASSIFICATIONS:
        assert entry.resolves_owner_entrypoint(), entry.surface_id


def test_full_chaser_profile_is_covered_exactly_without_promoting_components() -> None:
    profile = load_chaser_analysis_profile(full_chaser_analysis_profile_path())
    modules = {module.module_id: module for module in profile.modules}
    assert set(modules) == CHASER_PROFILE_SURFACE_IDS

    for module_id, module in modules.items():
        entry = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID[module_id]
        assert entry.owner_module == module.implementation
        assert (entry.schema_id, entry.schema_version) == (
            module.schema_id,
            module.schema_version,
        )
        implementation = importlib.import_module(module.implementation)
        assert implementation.SCHEMA_ID == entry.schema_id
        assert implementation.SCHEMA_VERSION == entry.schema_version
        if module_id in {"stimulus_epochs", "detection_occupancy", "chaser_distance"}:
            assert (
                entry.classification
                is AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY
            )
            assert entry.central_storage_catalog_required is True
        else:
            assert entry.classification is AnalyticsSurfaceClass.EMBEDDED_COMPONENT
            assert entry.central_storage_catalog_required is False
            assert entry.exact_storage_contract_required is True
            assert entry.owner_path.startswith("analysis/chaser_distance_runs/{run}/")
            assert implementation.COMPONENT_PARENT_NAME in entry.owner_path


def test_current_top_level_catalog_gaps_are_explicit_not_false_adoptions() -> None:
    expected = {
        "stimulus_epochs",
        "detection_occupancy",
        "session_occupancy",
        "chaser_distance",
    }
    actual = {
        entry.surface_id
        for entry in ANALYTICS_SURFACE_CLASSIFICATIONS
        if entry.central_storage_catalog_required
    }
    assert actual == expected
    assert actual.isdisjoint(DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE)


def test_registered_gate_publication_is_hardened_while_storage_contract_is_open() -> (
    None
):
    gate = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["registered_detection_gate"]
    audit = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["registered_detection_gate_audit"]
    assert gate.classification is AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY
    assert gate.owner_path == "analysis/detection_gate_runs/{run}"
    assert gate.central_storage_catalog_required is False
    assert gate.exact_storage_contract_required is True
    assert gate.exact_storage_contract_status is ExactStorageContractStatus.REQUIRED
    assert gate.lifecycle_context is AnalyticsLifecycleContext.POLICY_EVIDENCE_RUN
    assert gate.consumer_scope is AnalyticsConsumerScope.REGISTERED_DETECTION_GATE
    assert audit.classification is AnalyticsSurfaceClass.EXPORT
    assert audit.array_bearing is False
    assert (
        audit.exact_storage_contract_status is ExactStorageContractStatus.NOT_APPLICABLE
    )
    assert audit.owner_path == "{output_dir}/audit_report.json"
    gate_module = importlib.import_module(gate.owner_module)
    audit_module = importlib.import_module(audit.owner_module)
    assert (gate_module.GATE_RUN_SCHEMA_ID, gate_module.GATE_RUN_SCHEMA_VERSION) == (
        gate.schema_id,
        gate.schema_version,
    )
    assert callable(gate_module.validate_registered_detection_gate_run)
    assert (audit_module.SCHEMA_ID, audit_module.SCHEMA_VERSION) == (
        audit.schema_id,
        audit.schema_version,
    )


def test_detection_quality_classifications_expose_both_current_storage_surfaces() -> (
    None
):
    collection = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["detection_quality_collection"]
    nested = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["detection_quality_nested_report"]
    stage = get_stage_spec("detect_quality")
    assert stage.artifact_families == ("detect_quality_runs",)
    assert stage_run_parent_paths("detect_quality") == ("detect_runs",)
    assert collection.owner_path == "detect_quality_runs/{run}"
    assert nested.owner_path == (
        "detect_runs/{detect_run}/quality_reports/{quality_run}"
    )
    assert (
        collection.exact_storage_contract_status is ExactStorageContractStatus.REQUIRED
    )
    assert nested.exact_storage_contract_status is ExactStorageContractStatus.REQUIRED
    assert collection.stage_binding == nested.stage_binding == "detect_quality"
    assert (
        collection.lifecycle_context
        is AnalyticsLifecycleContext.RECORDING_WIDE_QUALITY_SNAPSHOT
    )
    assert (
        collection.consumer_scope
        is AnalyticsConsumerScope.RECORDING_WIDE_DETECTION_QUALITY
    )
    assert (
        nested.lifecycle_context
        is AnalyticsLifecycleContext.SOURCE_DETECT_NESTED_QUALITY_REPORT
    )
    assert nested.consumer_scope is AnalyticsConsumerScope.SOURCE_DETECT_LOCAL_QUALITY
    assert collection.central_storage_catalog_required is False
    assert nested.central_storage_catalog_required is False
    collection_module = importlib.import_module(collection.owner_module)
    assert collection_module.COLLECTION_QUALITY_SCHEMA == collection.schema_id
    refine_module = importlib.import_module("fisheye.refinement.refine_detect")
    registry_module = importlib.import_module("fisheye.registry.maintenance")
    assert callable(refine_module._resolve_modern_quality_group)
    assert callable(refine_module._resolve_detection_quality_labels)
    assert callable(registry_module._resolve_detect_quality_group)


def test_legacy_and_in_place_surfaces_cannot_be_mistaken_for_current_authorities() -> (
    None
):
    speed = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["speed_runs"]
    interpolation = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["chaser_state_interpolation"]
    assert speed.classification is AnalyticsSurfaceClass.LEGACY
    assert speed.lifecycle is AnalyticsSurfaceLifecycle.LEGACY
    assert speed.central_storage_catalog_required is False
    assert speed.exact_storage_contract_required is False
    assert (
        speed.exact_storage_contract_status
        is ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
    )
    assert interpolation.classification is AnalyticsSurfaceClass.MAINTENANCE_OUTPUT
    assert interpolation.lifecycle is AnalyticsSurfaceLifecycle.LEGACY
    assert interpolation.mutation_mode is AnalyticsMutationMode.IN_PLACE_MUTATION
    assert (
        interpolation.exact_storage_contract_status
        is ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
    )
    assert interpolation.owner_path == "analysis/stimulus_runs/{run}"


def test_active_swim_bout_statistics_writer_is_an_explicit_namespace_collision() -> (
    None
):
    statistics = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["swim_bout_statistics"]
    maintained = DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE["swim_bouts"]
    assert statistics.classification is AnalyticsSurfaceClass.LEGACY
    assert statistics.lifecycle is AnalyticsSurfaceLifecycle.CURRENT_LEGACY_SHAPED
    assert (
        statistics.lifecycle_context
        is AnalyticsLifecycleContext.ACTIVE_LEGACY_NAMESPACE_COLLISION
    )
    assert statistics.consumer_scope is AnalyticsConsumerScope.SHARED_SWIM_BOUT_SELECTOR
    assert statistics.owner_path == f"{maintained.run_parent}/{{run}}"
    assert statistics.stage_binding == "swim_bout_statistics"
    assert statistics.schema_id is None
    assert (
        statistics.exact_storage_contract_status is ExactStorageContractStatus.REQUIRED
    )
    assert statistics.central_storage_catalog_required is False
    assert statistics.surface_id not in DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE
    writer = importlib.import_module("fisheye.analysis.swim_bout_statistics")
    publication_source = inspect.getsource(writer._save_report_to_zarr)
    assert "stage_selector_eligible" not in publication_source
    assert "activate_selector_eligible_run" not in publication_source


def test_track_visualization_agrees_with_stage_and_run_parent_catalogs() -> None:
    entry = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["track_kinematics_visualization"]
    stage = get_stage_spec(entry.surface_id)
    assert stage.artifact_families == (entry.owner_path,)
    assert stage_run_parent_paths(
        entry.surface_id, artifact_families=stage.artifact_families
    ) == (entry.owner_path,)
    assert entry.classification is AnalyticsSurfaceClass.VISUALIZATION_CACHE
    assert entry.array_bearing is True
    assert entry.central_storage_catalog_required is False
    assert entry.exact_storage_contract_required is False
    assert (
        entry.exact_storage_contract_status is ExactStorageContractStatus.NOT_APPLICABLE
    )


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"surface_id": "bad/path"}, "canonical identifier"),
        ({"classification": "legacy"}, "closed enum"),
        ({"lifecycle_context": "top_level_run"}, "closed enum"),
        ({"consumer_scope": "recording_analysis"}, "closed enum"),
        ({"array_bearing": 1}, "exact bool"),
        ({"exact_storage_contract_status": "required"}, "closed enum"),
        ({"owner_module": "analysis.owner"}, "exact fisheye module"),
        ({"stage_binding": "bad/stage"}, "canonical identifier"),
        ({"owner_path": "/analysis/runs"}, "canonical path pattern"),
        ({"schema_id": "palette.unpaired"}, "declared together"),
        (
            {"central_storage_catalog_required": True},
            "central catalog adoption",
        ),
    ),
)
def test_classification_rejects_contradictory_declarations(
    changes: dict[str, object], error: str
) -> None:
    base = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["speed_runs"]
    with pytest.raises((TypeError, ValueError), match=error):
        replace(base, **changes)


def test_in_place_mutation_is_reserved_for_maintenance() -> None:
    base = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["speed_runs"]
    with pytest.raises(ValueError, match="reserved for maintenance"):
        replace(base, mutation_mode=AnalyticsMutationMode.IN_PLACE_MUTATION)


@pytest.mark.parametrize("owner_path", ("analysis/bad\\path", "analysis/bad\npath"))
def test_owner_path_rejects_backslash_and_control_characters(owner_path: str) -> None:
    base = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["speed_runs"]
    with pytest.raises(ValueError, match="canonical path pattern"):
        replace(base, owner_path=owner_path)


@pytest.mark.parametrize(
    "owner_path",
    (
        "analysis/{run",
        "analysis/run}",
        "analysis/{}",
        "analysis/{bad-name}",
        "analysis/{run!r}",
        "analysis/{run:>8}",
    ),
)
def test_owner_path_rejects_malformed_or_noncanonical_templates(
    owner_path: str,
) -> None:
    base = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["speed_runs"]
    with pytest.raises(ValueError, match="owner_path"):
        replace(base, owner_path=owner_path)


@pytest.mark.parametrize(
    "schema_id",
    (
        " palette.bad",
        "palette.bad ",
        "Palette.bad",
        "palette.bad id",
        "palette.bad-name",
        "palette..bad",
        "palette.bad/child",
        "palette.",
    ),
)
def test_schema_identity_rejects_whitespace_and_noncanonical_ids(
    schema_id: str,
) -> None:
    base = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["registered_detection_gate"]
    with pytest.raises(ValueError, match="schema identity"):
        replace(base, schema_id=schema_id)


@pytest.mark.parametrize(
    ("surface_id", "changes", "error"),
    (
        (
            "registered_detection_gate",
            {"lifecycle": AnalyticsSurfaceLifecycle.LEGACY},
            "scientific authorities and components must be current",
        ),
        (
            "registered_detection_gate",
            {
                "exact_storage_contract_status": (
                    ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
                )
            },
            "current surfaces cannot be legacy compatibility-only",
        ),
        (
            "registered_detection_gate",
            {
                "exact_storage_contract_status": ExactStorageContractStatus.NOT_APPLICABLE
            },
            "require an exact storage-contract disposition",
        ),
        (
            "speed_runs",
            {"lifecycle": AnalyticsSurfaceLifecycle.CURRENT},
            "explicit legacy lifecycle",
        ),
        (
            "swim_bout_statistics",
            {"lifecycle": AnalyticsSurfaceLifecycle.LEGACY},
            "required exact contracts",
        ),
        (
            "swim_bout_statistics",
            {
                "exact_storage_contract_status": (
                    ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
                )
            },
            "active namespace collisions",
        ),
        (
            "registered_detection_gate_audit",
            {"array_bearing": True},
            "agree with lifecycle_context",
        ),
        (
            "chaser_quadrant_occupancy",
            {"central_storage_catalog_required": True},
            "central catalog adoption",
        ),
        (
            "stimulus_epochs",
            {"consumer_scope": AnalyticsConsumerScope.CHASER_COMPONENT_READER},
            "agree with lifecycle_context",
        ),
        (
            "speed_runs",
            {"exact_storage_contract_status": ExactStorageContractStatus.REQUIRED},
            "required exact contracts",
        ),
        (
            "registered_detection_gate_audit",
            {"exact_storage_contract_status": ExactStorageContractStatus.REQUIRED},
            "required exact contracts",
        ),
        (
            "detection_quality_collection",
            {"owner_path": "detect_runs/{run}/quality_reports/{quality_run}"},
            "recording-wide quality context",
        ),
        (
            "detection_quality_nested_report",
            {"stage_binding": "nested_quality"},
            "source-local quality context",
        ),
    ),
)
def test_semantic_closure_rejects_impossible_cross_field_combinations(
    surface_id: str,
    changes: dict[str, object],
    error: str,
) -> None:
    base = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID[surface_id]
    with pytest.raises((TypeError, ValueError), match=error):
        replace(base, **changes)
