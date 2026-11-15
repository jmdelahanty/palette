"""LabelingStore.task_row_progress: per-task row counts from the store alone."""

from __future__ import annotations

from fisheye.labeling.assignment_store import CARRY_FORWARD_SESSION_CLIENT_LABEL, LabelingStore


def _checkpoint(store: LabelingStore, session_id: str, task_id: str, roi_idx: int) -> dict[str, object]:
    return store.upsert_session_checkpoint(
        session_id=session_id, task_id=task_id, recording_id="rec", user="u",
        workflow_kind="keypoints", target_run_path="refined_keypoints_runs/r",
        target_edit_revision=0, source_rowset_path=None, roi_idx=roi_idx,
        component_name="keypoints", payload={},
    )


def _apply_all(store: LabelingStore, task_id: str, apply_id: str) -> None:
    claimed = store.claim_session_checkpoints_for_apply(
        task_id=task_id, component_name="keypoints", apply_id=apply_id
    )
    store.mark_session_checkpoints_applied(
        checkpoint_ids=[c["checkpoint_id"] for c in claimed], apply_id=apply_id,
        edit_revision_before=0, edit_revision_after=1,
    )


def _store(tmp_path) -> LabelingStore:
    store = LabelingStore(tmp_path / "labeling.sqlite")
    store.initialize()
    store.assign_recording(recording_id="rec", assignee_user="u", assigned_by="u")
    store.upsert_task(recording_id="rec", task_id="targeted", workflow_kind="keypoints",
                      scope={"target_roi_indices": [0, 1, 2]})
    store.upsert_task(recording_id="rec", task_id="all-rows", workflow_kind="keypoints",
                      scope={"include_all": True})
    return store


def test_targeted_task_counts_only_target_rows_by_state(tmp_path):
    store = _store(tmp_path)
    try:
        session = store.create_session(task_id="targeted", user="u").session_id
        _checkpoint(store, session, "targeted", 0)
        _apply_all(store, "targeted", "apply-1")
        _checkpoint(store, session, "targeted", 1)
        _checkpoint(store, session, "targeted", 5)  # outside the task's targets
        discarded = _checkpoint(store, session, "targeted", 2)
        store.discard_session_checkpoints(
            task_id="targeted", checkpoint_ids=[discarded["checkpoint_id"]], user="u", reason="test"
        )
        assert store.task_row_progress(["targeted"]) == {
            "targeted": {
                "row_total": 3,
                "saved_row_count": 2,
                "applied_row_count": 1,
                "unapplied_row_count": 1,
                "carried_row_count": 0,
            }
        }
    finally:
        store.close()


def test_all_rows_task_has_unknown_total_and_counts_carried_rows(tmp_path):
    store = _store(tmp_path)
    try:
        carry = store.create_session(
            task_id="all-rows", user="u", client_label=CARRY_FORWARD_SESSION_CLIENT_LABEL
        ).session_id
        _checkpoint(store, carry, "all-rows", 0)
        _checkpoint(store, carry, "all-rows", 1)
        _apply_all(store, "all-rows", "carry-apply")
        store.close_session(session_id=carry, user="u")
        labeler = store.create_session(task_id="all-rows", user="u", client_label="browser").session_id
        # Re-editing a carried row rewrites its one checkpoint (unique per task,
        # row and component) under the labeler's session: it is no longer carried.
        _checkpoint(store, labeler, "all-rows", 1)
        _checkpoint(store, labeler, "all-rows", 7)
        progress = store.task_row_progress(["all-rows", "missing-task"])
        assert progress == {
            "all-rows": {
                "row_total": None,
                "saved_row_count": 3,
                "applied_row_count": 1,
                "unapplied_row_count": 2,
                "carried_row_count": 1,
            }
        }
        assert store.task_row_progress([]) == {}
    finally:
        store.close()


def test_carry_forward_tool_labels_its_session_with_the_shared_constant():
    import inspect

    from fisheye.labeling import carry_forward_tail_keypoints

    source = inspect.getsource(carry_forward_tail_keypoints)
    assert "client_label=CARRY_FORWARD_SESSION_CLIENT_LABEL" in source
    assert '"carry_forward_tail_keypoints"' not in source
