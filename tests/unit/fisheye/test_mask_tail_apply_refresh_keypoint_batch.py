"""Carry-forward, batch keypoint Apply, task states, and v3/v2 parity."""

from tests.unit.fisheye.mask_tail_apply_refresh_cases import (  # noqa: F401
    reviewed_archive,
    successor_format,
    test_batch_keypoint_apply_writes_every_row_of_a_successor_version,
    test_carry_forward_moves_stranded_edit_into_newest_version,
    test_discard_marks_only_unapplied_checkpoints_and_audits,
    test_interrupted_batch_apply_fails_closed_under_the_archive_lock,
    test_source_change_during_publication_refuses_stale_snapshot,
    test_task_state_domain_is_enforced_and_reported,
    test_v3_matches_v2_exactly_on_rows_legacy_derives,
)


__all__ = [
    "reviewed_archive",
    "successor_format",
    "test_batch_keypoint_apply_writes_every_row_of_a_successor_version",
    "test_carry_forward_moves_stranded_edit_into_newest_version",
    "test_discard_marks_only_unapplied_checkpoints_and_audits",
    "test_interrupted_batch_apply_fails_closed_under_the_archive_lock",
    "test_source_change_during_publication_refuses_stale_snapshot",
    "test_task_state_domain_is_enforced_and_reported",
    "test_v3_matches_v2_exactly_on_rows_legacy_derives",
]
