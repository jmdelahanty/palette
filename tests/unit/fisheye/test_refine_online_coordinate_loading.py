"""Loading and numerical cases for the refine-online coordinate contract."""

from tests.unit.fisheye.refine_online_coordinate_contract_cases import (
    test_duplicate_acquisition_time_fails_even_when_camera_ids_are_unique,
    test_load_selects_chaser_by_stimulus_key_not_external_camera_id,
    test_load_uses_exact_child_surface_and_stimulus_identity,
    test_normal_path_rejects_noncanonical_archive_without_legacy_adapter,
    test_smoothing_never_crosses_nonconsecutive_acquisition_frames,
    test_source_authority_tamper_fails_before_output_creation,
    test_stimulus_rows_are_ordered_by_acquisition_not_external_camera_id,
)


__all__ = [
    "test_duplicate_acquisition_time_fails_even_when_camera_ids_are_unique",
    "test_load_selects_chaser_by_stimulus_key_not_external_camera_id",
    "test_load_uses_exact_child_surface_and_stimulus_identity",
    "test_normal_path_rejects_noncanonical_archive_without_legacy_adapter",
    "test_smoothing_never_crosses_nonconsecutive_acquisition_frames",
    "test_source_authority_tamper_fails_before_output_creation",
    "test_stimulus_rows_are_ordered_by_acquisition_not_external_camera_id",
]
