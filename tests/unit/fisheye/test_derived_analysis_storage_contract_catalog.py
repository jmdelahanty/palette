from fisheye.analysis_workflows.storage_contract_catalog import (
    DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS,
    DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE,
    DERIVED_ANALYSIS_STORAGE_CONTRACTS,
    SERIALIZED_REGISTRY_STAGE_IDS,
    resolved_storage_contracts,
)
from fisheye.registry.stage_catalog import get_stage_spec
from fisheye.shared.stage_run_groups import stage_run_parent_paths


EXPECTED_SCHEMA_IDENTITIES = {
    "track_kinematics": ("analysis.track_kinematics_runs", 1),
    "swim_bouts": ("palette.swim_bout_runs", 8),
    "bout_kinematics": ("analysis.bout_kinematics_runs", 7),
    "eye_angles": ("analysis.eye_angle_runs", 6),
    "subject_shape": ("analysis.subject_shape_runs", 4),
    "tail_kinematics": ("analysis.tail_kinematics_runs", 2),
    "stimulus_response": ("palette.stimulus_response", 2),
}


def test_catalog_has_one_contract_per_maintained_stage_and_parent() -> None:
    assert set(DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE) == set(
        EXPECTED_SCHEMA_IDENTITIES
    )
    assert len(DERIVED_ANALYSIS_STORAGE_CONTRACTS) == len(
        DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE
    )
    parents = [contract.run_parent for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS]
    assert len(parents) == len(set(parents))


def test_catalog_resolves_live_writer_schema_constants() -> None:
    resolved = {
        contract["stage_id"]: contract for contract in resolved_storage_contracts()
    }
    assert {
        stage: (record["schema_id"], record["schema_version"])
        for stage, record in resolved.items()
    } == EXPECTED_SCHEMA_IDENTITIES


def test_catalog_run_parents_agree_with_registry_stage_catalog() -> None:
    for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS:
        stage = get_stage_spec(contract.stage_id)
        assert contract.run_parent in stage.artifact_families
        assert contract.run_parent in stage_run_parent_paths(contract.stage_id)


def test_availability_parents_are_derived_from_the_storage_catalog() -> None:
    for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS:
        assert DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS[contract.stage_id] == (
            contract.availability_parents
        )
        assert contract.availability_parents
        assert all(
            parent == contract.run_parent
            or parent.startswith(f"{contract.run_parent}/")
            for parent in contract.availability_parents
        )


def test_every_cataloged_materializer_uses_shared_atomic_publisher() -> None:
    for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS:
        assert contract.uses_shared_atomic_publisher(), contract.stage_id


def test_byte_planner_migration_boundary_is_explicit() -> None:
    for record in resolved_storage_contracts():
        assert isinstance(record["byte_planner_adopted"], bool)
        assert record["physical_policy_owner"]


def test_serialized_registry_scope_is_explicit_and_narrow() -> None:
    assert SERIALIZED_REGISTRY_STAGE_IDS == {"eye_angles", "track_kinematics"}
    for record in resolved_storage_contracts():
        expected = (
            "serialized_finalizer_v1"
            if record["stage_id"] in SERIALIZED_REGISTRY_STAGE_IDS
            else "not_implemented"
        )
        assert record["registry_publication"] == expected
