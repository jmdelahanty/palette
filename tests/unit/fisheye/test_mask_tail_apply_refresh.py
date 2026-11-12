"""Successor versioning and explicit geometry upgrades."""

# ``browser_context`` and ``reviewed_archive`` stay importable here for the
# sibling labeling suites that reuse them.
from tests.unit.fisheye.mask_tail_apply_refresh_cases import (  # noqa: F401
    browser_context,
    reviewed_archive,
    successor_format,
    test_explicit_geometry_upgrade_keeps_other_seed_rows_and_manual_clears,
    test_geometry_upgrade_refuses_nonfailure_and_records_changed_non_target_mask,
    test_geometry_upgrade_refuses_noninteger_target_rows,
    test_geometry_upgrade_refuses_tampered_immutable_seed,
    test_later_apply_preserves_mixed_row_methods_from_real_curled_mask,
    test_manual_clear_remains_cleared_and_ineligible,
    test_new_version_keeps_manual_points_failures_and_source_history,
    test_same_source_retry_reuses_version_and_preserves_successor_edits,
)


__all__ = [
    "browser_context",
    "reviewed_archive",
    "successor_format",
    "test_explicit_geometry_upgrade_keeps_other_seed_rows_and_manual_clears",
    "test_geometry_upgrade_refuses_nonfailure_and_records_changed_non_target_mask",
    "test_geometry_upgrade_refuses_noninteger_target_rows",
    "test_geometry_upgrade_refuses_tampered_immutable_seed",
    "test_later_apply_preserves_mixed_row_methods_from_real_curled_mask",
    "test_manual_clear_remains_cleared_and_ineligible",
    "test_new_version_keeps_manual_points_failures_and_source_history",
    "test_same_source_retry_reuses_version_and_preserves_successor_edits",
]
