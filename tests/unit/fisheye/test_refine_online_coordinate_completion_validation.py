"""Completion-validation cases for the refine-online coordinate contract."""

from tests.unit.fisheye.refine_online_coordinate_contract_cases import (
    test_completed_refinement_fixture_clones_are_isolated,
    test_completion_rejects_identity_acquisition_and_surface_drift,
    test_completion_rejects_missing_lineage_or_unsupported_space,
    test_completion_rejects_processing_lineage_drift,
    test_completion_rejects_wrong_transform_order,
    test_public_coordinate_loaders_cannot_read_staging_evidence,
)


__all__ = [
    "test_completed_refinement_fixture_clones_are_isolated",
    "test_completion_rejects_identity_acquisition_and_surface_drift",
    "test_completion_rejects_missing_lineage_or_unsupported_space",
    "test_completion_rejects_processing_lineage_drift",
    "test_completion_rejects_wrong_transform_order",
    "test_public_coordinate_loaders_cannot_read_staging_evidence",
]
