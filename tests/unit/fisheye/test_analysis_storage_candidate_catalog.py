from __future__ import annotations

from dataclasses import replace

import pytest

from fisheye.analysis_workflows.storage_candidate_catalog import (
    DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE,
    DERIVED_ANALYSIS_STORAGE_CANDIDATES,
    StorageCandidatePublicationMode,
    resolved_storage_candidates,
)
from fisheye.analysis_workflows.storage_contract_catalog import (
    DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE,
)


EXPECTED_ATOMIC = {
    "track_kinematics",
    "swim_bouts",
    "bout_kinematics",
    "eye_angles",
    "subject_shape",
    "tail_kinematics",
    "stimulus_response",
    "stimulus_epochs",
    "detection_occupancy",
    "session_occupancy",
}
EXPECTED_DIRECT = {"tail_posture_view", "bout_classification"}


def test_candidate_catalog_is_closed_executable_and_unpromoted() -> None:
    assert set(DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE) == (
        EXPECTED_ATOMIC | EXPECTED_DIRECT
    )
    assert len(DERIVED_ANALYSIS_STORAGE_CANDIDATES) == len(
        DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE
    )
    assert set(DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE) == set(
        DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE
    )
    for candidate in DERIVED_ANALYSIS_STORAGE_CANDIDATES:
        assert candidate.resolves_entrypoint(), candidate.stage_id
        assert candidate.run_parent == (
            DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE[candidate.stage_id].run_parent
        )
        record = candidate.as_record()
        assert record["selector_eligible"] is False
        assert record["profile_promoted"] is False


def test_atomic_and_guarded_direct_boundaries_are_exact() -> None:
    for stage_id in EXPECTED_ATOMIC:
        candidate = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE[stage_id]
        assert candidate.publication_mode is StorageCandidatePublicationMode.SHARED_ATOMIC
        assert candidate.consolidates_before_return is True
        assert candidate.repairs_failed_visibility is True
        assert candidate.uses_shared_atomic_publisher()
    for stage_id in EXPECTED_DIRECT:
        candidate = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE[stage_id]
        assert candidate.publication_mode is StorageCandidatePublicationMode.GUARDED_DIRECT
        assert candidate.consolidates_before_return is False
        assert candidate.repairs_failed_visibility is False
        assert not candidate.uses_shared_atomic_publisher()


def test_resolved_candidate_records_are_stable_and_json_compatible() -> None:
    records = resolved_storage_candidates()
    assert [record["stage_id"] for record in records] == [
        candidate.stage_id for candidate in DERIVED_ANALYSIS_STORAGE_CANDIDATES
    ]
    assert all(isinstance(record["profile_id"], str) for record in records)


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"stage_id": "unknown_stage"}, "central logical contract"),
        ({"profile_id": "bad profile"}, "canonical exact string"),
        ({"run_parent": "analysis/bad path"}, "canonical relative path"),
        ({"owner_module": "not a module"}, "canonical exact string"),
        ({"entrypoint_attr": "write-run"}, "canonical exact string"),
        ({"publication_mode": "atomic"}, "publication_mode"),
        ({"consolidates_before_return": 1}, "exact bool"),
        ({"repairs_failed_visibility": 1}, "exact bool"),
        (
            {"repairs_failed_visibility": True, "consolidates_before_return": False},
            "requires candidate consolidation",
        ),
    ),
)
def test_candidate_declaration_fails_closed(
    changes: dict[str, object],
    error: str,
) -> None:
    base = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE["eye_angles"]
    with pytest.raises((TypeError, ValueError), match=error):
        replace(base, **changes)


def test_guarded_direct_candidate_cannot_claim_atomic_guarantees() -> None:
    base = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE["tail_posture_view"]
    with pytest.raises(ValueError, match="do not own archive consolidation"):
        replace(base, consolidates_before_return=True)


def test_shared_atomic_candidate_requires_consolidation_and_repair() -> None:
    base = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE["eye_angles"]
    with pytest.raises(ValueError, match="must consolidate and repair"):
        replace(base, repairs_failed_visibility=False)
