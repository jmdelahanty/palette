"""Browser successor offers and labeling-task lifecycle."""

from tests.unit.fisheye.mask_tail_apply_refresh_cases import (  # noqa: F401
    reviewed_archive,
    successor_format,
    test_browser_offers_successor_without_resetting_source_or_opened_successor,
    test_completed_browser_effect_refuses_damaged_publication,
    test_completing_mask_review_retires_untouched_mask_successor,
    test_failed_successor_publication_restores_superseded_tasks,
    test_pending_paired_pose_checkpoint_blocks_successor_publication,
    test_stranded_older_version_edit_blocks_mask_apply,
    test_superseded_task_refuses_new_edits_and_is_hidden,
)


__all__ = [
    "reviewed_archive",
    "successor_format",
    "test_browser_offers_successor_without_resetting_source_or_opened_successor",
    "test_completed_browser_effect_refuses_damaged_publication",
    "test_completing_mask_review_retires_untouched_mask_successor",
    "test_failed_successor_publication_restores_superseded_tasks",
    "test_pending_paired_pose_checkpoint_blocks_successor_publication",
    "test_stranded_older_version_edit_blocks_mask_apply",
    "test_superseded_task_refuses_new_edits_and_is_hidden",
]
