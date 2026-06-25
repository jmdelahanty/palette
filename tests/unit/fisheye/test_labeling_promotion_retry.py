from __future__ import annotations

from fisheye.labeling.assignment_store import LabelingStore


def _store_with_failed_promotion(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    store.initialize()
    store.assign_recording(recording_id="rec-a", assignee_user="alice")
    store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_analysis")
    failed = store.record_event(
        task_id="task-a",
        recording_id="rec-a",
        user="alice",
        event_type="promotion_failed",
        target={"source_frame_index": 12, "training_zarr": "/tmp/training.zarr"},
        after={"error": "promotion target missing"},
    )
    return store, failed


def test_promotion_retry_claim_blocks_duplicate_in_flight_retry(tmp_path):
    store, failed = _store_with_failed_promotion(tmp_path)
    try:
        claim = store.claim_promotion_retry(
            failed_event_id=str(failed["event_id"]),
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
        )
        duplicate = store.claim_promotion_retry(
            failed_event_id=str(failed["event_id"]),
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
        )

        assert claim["status"] == "claimed"
        assert claim["event"]["event_type"] == "promotion_retry_started"
        assert duplicate["status"] == "in_progress"
        assert duplicate["event"]["event_id"] == claim["event"]["event_id"]
    finally:
        store.close()

def test_promotion_retry_can_be_reclaimed_after_failed_retry(tmp_path):
    store, failed = _store_with_failed_promotion(tmp_path)
    try:
        first = store.claim_promotion_retry(
            failed_event_id=str(failed["event_id"]),
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
        )
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="promotion_failed",
            target={"retry_of_event_id": str(failed["event_id"])},
            after={"error": "still missing"},
        )

        second = store.claim_promotion_retry(
            failed_event_id=str(failed["event_id"]),
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
        )

        assert first["status"] == "claimed"
        assert second["status"] == "claimed"
        assert second["event"]["event_id"] != first["event"]["event_id"]
    finally:
        store.close()


def test_promotion_success_makes_retry_idempotent_and_hides_failed_event(tmp_path):
    store, failed = _store_with_failed_promotion(tmp_path)
    try:
        store.claim_promotion_retry(
            failed_event_id=str(failed["event_id"]),
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
        )
        success = store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="promotion_success",
            target={"retry_of_event_id": str(failed["event_id"]), "training_zarr": "/tmp/training.zarr"},
            after={"promoted_count": 1},
        )

        duplicate = store.claim_promotion_retry(
            failed_event_id=str(failed["event_id"]),
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
        )
        summary = store.task_summary_for_user("alice")

        assert duplicate["status"] == "already_succeeded"
        assert duplicate["event"]["event_id"] == success["event_id"]
        assert summary["failed_promotion_count"] == 0
        assert summary["failed_promotions"] == []
    finally:
        store.close()
