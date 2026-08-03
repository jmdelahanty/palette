from dataclasses import replace
import importlib
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


def test_registered_gate_uses_its_independent_exact_publication_contract() -> None:
    gate = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["registered_detection_gate"]
    audit = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["registered_detection_gate_audit"]
    assert gate.classification is AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY
    assert gate.owner_path == "analysis/detection_gate_runs/{run}"
    assert gate.central_storage_catalog_required is False
    assert gate.exact_storage_contract_required is False
    assert (
        gate.exact_storage_contract_status
        is ExactStorageContractStatus.IMPLEMENTED_INDEPENDENT
    )
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
        collection.exact_storage_contract_status
        is ExactStorageContractStatus.IMPLEMENTED_INDEPENDENT
    )
    assert nested.exact_storage_contract_status is ExactStorageContractStatus.REQUIRED
    assert collection.central_storage_catalog_required is False
    assert nested.central_storage_catalog_required is False
    collection_module = importlib.import_module(collection.owner_module)
    assert collection_module.COLLECTION_QUALITY_SCHEMA == collection.schema_id


def test_legacy_and_in_place_surfaces_cannot_be_mistaken_for_current_authorities() -> (
    None
):
    speed = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["speed_runs"]
    statistics = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["swim_bout_statistics"]
    interpolation = ANALYTICS_SURFACE_CLASSIFICATION_BY_ID["chaser_state_interpolation"]
    for entry in (speed, statistics):
        assert entry.classification is AnalyticsSurfaceClass.LEGACY
        assert entry.lifecycle is AnalyticsSurfaceLifecycle.LEGACY
        assert entry.central_storage_catalog_required is False
        assert entry.exact_storage_contract_required is False
        assert (
            entry.exact_storage_contract_status
            is ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
        )
    assert statistics.owner_path == "analysis/swim_bout_runs/{run}"
    assert interpolation.classification is AnalyticsSurfaceClass.MAINTENANCE_OUTPUT
    assert interpolation.lifecycle is AnalyticsSurfaceLifecycle.LEGACY
    assert interpolation.mutation_mode is AnalyticsMutationMode.IN_PLACE_MUTATION
    assert (
        interpolation.exact_storage_contract_status
        is ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
    )
    assert interpolation.owner_path == "analysis/stimulus_runs/{run}"


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
        ({"array_bearing": 1}, "exact bool"),
        ({"exact_storage_contract_status": "required"}, "closed enum"),
        ({"owner_module": "analysis.owner"}, "exact fisheye module"),
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
