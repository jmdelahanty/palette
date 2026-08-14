"""Lifecycle guard cases for the refine-online coordinate contract."""

from tests.unit.fisheye.refine_online_coordinate_contract_cases import (
    test_failure_immediately_after_public_creation_retains_owned_tombstone,
    test_occupied_pending_is_refused_before_candidate_creation,
    test_run_name_collision_never_deletes_a_preexisting_run,
)


__all__ = [
    "test_failure_immediately_after_public_creation_retains_owned_tombstone",
    "test_occupied_pending_is_refused_before_candidate_creation",
    "test_run_name_collision_never_deletes_a_preexisting_run",
]
