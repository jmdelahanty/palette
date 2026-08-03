from dataclasses import replace

import pytest

from fisheye.analysis_workflows.storage_contract_catalog import (
    DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS,
    DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE,
    DERIVED_ANALYSIS_STORAGE_CONTRACTS,
    SERIALIZED_REGISTRY_STAGE_IDS,
    resolved_storage_contracts,
)
from fisheye.analysis_workflows.availability import (
    STAGE_RUN_PARENTS as AVAILABILITY_STAGE_RUN_PARENTS,
)
from fisheye.registry.stage_catalog import get_stage_spec
from fisheye.shared.stage_run_groups import stage_run_parent_paths


EXPECTED_SCHEMA_IDENTITIES = {
    "track_kinematics": ("analysis.track_kinematics_runs", 1),
    "swim_bouts": ("palette.swim_bout_runs", 8),
    "bout_kinematics": ("analysis.bout_kinematics_runs", 7),
    "eye_angles": ("analysis.eye_angle_runs", 7),
    "subject_shape": ("analysis.subject_shape_runs", 4),
    "tail_kinematics": ("analysis.tail_kinematics_runs", 2),
    "tail_posture_view": ("analysis.tail_posture_view_runs", 3),
    "bout_classification": ("analysis.bout_classification_runs", 2),
    "stimulus_response": ("palette.stimulus_response", 2),
}

EXPECTED_DIRECT_WRITERS = {
    "tail_posture_view": (
        "fisheye.analysis.tail_posture_view_runs",
        "write_tail_posture_view_run",
        "tail_posture_view_from_subject_shape",
        "refined_subject_mask_metric_row_chunk_compatibility",
    ),
    "bout_classification": (
        "fisheye.analysis.megabouts_classifier",
        "write_megabouts_classification_run",
        "palette_megabouts_direct_classifier",
        "columnar_store_array_v1",
    ),
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
        assert stage.artifact_families == (contract.run_parent,)
        assert stage_run_parent_paths(contract.stage_id) == (contract.run_parent,)


def test_availability_parents_are_derived_from_the_storage_catalog() -> None:
    for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS:
        assert DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS[contract.stage_id] == (
            contract.availability_parents
        )
        assert AVAILABILITY_STAGE_RUN_PARENTS[contract.stage_id] == (
            contract.availability_parents
        )
        assert contract.availability_parents
        assert all(
            parent == contract.run_parent
            or parent.startswith(f"{contract.run_parent}/")
            for parent in contract.availability_parents
        )


def test_publication_ownership_is_explicit_and_executable() -> None:
    for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS:
        if contract.publication_owner_kind == "shared_atomic_materializer_v1":
            assert contract.uses_shared_atomic_publisher(), contract.stage_id
            assert contract.publication_entrypoint_attr is None
        else:
            assert contract.publication_owner_kind == "guarded_direct_writer_v1"
            assert not contract.uses_shared_atomic_publisher(), contract.stage_id
            assert contract.resolves_publication_entrypoint(), contract.stage_id


def test_direct_writer_ownership_and_physical_policy_are_exact() -> None:
    resolved = {record["stage_id"]: record for record in resolved_storage_contracts()}
    for stage_id, (
        owner_module,
        entrypoint,
        method,
        physical_policy,
    ) in EXPECTED_DIRECT_WRITERS.items():
        contract = DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE[stage_id]
        record = resolved[stage_id]
        assert contract.materializer_module is None
        assert record["publication_owner_module"] == owner_module
        assert record["publication_entrypoint"] == entrypoint
        assert record["method"] == method
        assert record["physical_policy_owner"] == physical_policy
        assert record["registry_publication"] == "serialized_finalizer_v1"
        assert record["byte_planner_adopted"] is False


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"publication_owner_kind": "best_effort"}, "supported exact owner mode"),
        ({"materializer_module": None}, "requires materializer_module"),
        ({"publication_owner_module": "some.writer"}, "derives its owner"),
        ({"publication_entrypoint_attr": "write"}, "forbids a direct-writer"),
        ({"byte_planner_adopted": 1}, "exact bool"),
        ({"registry_publication": "eventually"}, "supported exact mode"),
        ({"stage_id": "analysis/eye_angles"}, "canonical identifier"),
        ({"stage_id": "bad stage"}, "canonical identifier"),
        ({"stage_id": "bad.stage"}, "canonical identifier"),
        ({"run_parent": "/analysis/runs"}, "canonical relative path"),
        ({"run_parent": "analysis/bad path"}, "canonical relative path"),
        ({"run_parent": "analysis\\bad"}, "canonical relative path"),
        ({"schema_module": "not a module"}, "canonical module path"),
        ({"schema_module": "fisheye..writer"}, "canonical module path"),
        ({"schema_id_attr": "bad attr"}, "canonical constant attr"),
        ({"schema_version_attr": "schema_version"}, "canonical constant attr"),
        ({"method_version_attr": "METHOD-VERSION"}, "canonical constant attr"),
        ({"layout_attr": "layout"}, "canonical constant attr"),
        ({"method_attr": "method"}, "canonical constant attr"),
        ({"physical_policy_owner": "policy name"}, "policy identifier"),
        (
            {"availability_parents": ("analysis/other_runs",)},
            "equal or be nested below",
        ),
        (
            {"availability_parents": ("analysis/eye_angle_runs/bad path",)},
            "canonical relative path",
        ),
    ),
)
def test_catalog_declaration_rejects_invalid_shared_owner_state(
    changes: dict[str, object],
    error: str,
) -> None:
    base = DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE["eye_angles"]
    with pytest.raises((TypeError, ValueError), match=error):
        replace(base, **changes)


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        (
            {"materializer_module": "fisheye.analysis.writer"},
            "must not claim a materializer",
        ),
        ({"publication_owner_module": None}, "requires an exact owner module"),
        ({"publication_owner_module": "not a module"}, "exact owner module"),
        ({"publication_entrypoint_attr": None}, "requires an exact entrypoint"),
        ({"publication_entrypoint_attr": "write-run"}, "exact entrypoint"),
    ),
)
def test_catalog_declaration_rejects_invalid_direct_owner_state(
    changes: dict[str, object],
    error: str,
) -> None:
    base = DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE["tail_posture_view"]
    with pytest.raises(ValueError, match=error):
        replace(base, **changes)


def test_byte_planner_migration_boundary_is_explicit() -> None:
    for record in resolved_storage_contracts():
        assert isinstance(record["byte_planner_adopted"], bool)
        assert record["physical_policy_owner"]


def test_serialized_registry_scope_covers_every_maintained_contract() -> None:
    assert SERIALIZED_REGISTRY_STAGE_IDS == set(EXPECTED_SCHEMA_IDENTITIES)
    for record in resolved_storage_contracts():
        assert record["registry_publication"] == "serialized_finalizer_v1"
