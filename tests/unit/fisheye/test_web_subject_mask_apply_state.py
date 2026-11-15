"""Receipt queries preserve run ownership and exact successor identities."""

from types import SimpleNamespace

import pytest

from fisheye.labeling.assignment_store import LabelingStore
from fisheye.labeling.web_subject_mask_apply_state import (
    TAIL_SUCCESSOR_EVENT,
    pending_mask_run_effects,
    require_mask_apply_ownership,
    tail_successor_offer,
)


def task(store, task_id, archive, component="subject_body"):
    store.upsert_task(
        task_id=task_id,
        recording_id="rec",
        workflow_kind="subject_mask_component",
        run_name="mask",
        component_name=component,
        scope={"zarr_path": str(archive), "refined_run": "mask"},
    )
    lease = store.create_session(task_id=task_id, user="reviewer")
    checkpoint = store.upsert_session_checkpoint(
        task_id=task_id,
        session_id=lease.session_id,
        recording_id="rec",
        user="reviewer",
        workflow_kind="subject_mask_component",
        target_run_path="refined_subject_masks_runs/mask",
        target_edit_revision=0,
        source_rowset_path="crop_runs/crop",
        roi_idx=0,
        component_name=component,
        payload={"mask": "test"},
        metadata={},
    )
    apply_id = "apply-" + task_id
    store.claim_session_checkpoints_for_apply(
        task_id=task_id, component_name=component, apply_id=apply_id
    )
    store.mark_session_checkpoints_applied(
        checkpoint_ids=[checkpoint["checkpoint_id"]],
        apply_id=apply_id,
        edit_revision_before=0,
        edit_revision_after=1,
        require_secondary_effects=True,
    )


def test_run_guard_includes_other_components_but_not_other_archives(tmp_path):
    store = LabelingStore(tmp_path / "tasks.sqlite")
    try:
        store.assign_recording(recording_id="rec", assignee_user="reviewer")
        archive = tmp_path / "a.zarr"
        task(store, "body", archive)
        task(store, "swim", archive, "swim_bladder")
        task(store, "other", tmp_path / "other" / "a.zarr")
        runtime = SimpleNamespace(
            zarr_path=str(archive),
            task_id="body",
            refined=SimpleNamespace(run_name="mask"),
        )
        pending = pending_mask_run_effects(store, runtime)
        assert {row["task_id"] for row in pending} == {"body", "swim"}
        with pytest.raises(RuntimeError, match="task swim"):
            require_mask_apply_ownership(store, runtime, "apply-body")
        store.mark_session_checkpoint_apply_effects_complete(
            task_id="swim", component_name="swim_bladder", apply_id="apply-swim"
        )
        require_mask_apply_ownership(store, runtime, "apply-body")
        with pytest.raises(RuntimeError, match="task body"):
            require_mask_apply_ownership(store, runtime, "different-apply")
        assert (
            store.count_pending_session_checkpoint_apply_effects(task_id="other") == 1
        )
    finally:
        store.close()


def test_exact_offer_lookup_survives_newer_events_and_rejects_stale_revision(tmp_path):
    store = LabelingStore(tmp_path / "tasks.sqlite")
    try:
        store.assign_recording(recording_id="rec", assignee_user="reviewer")
        store.upsert_task(
            task_id="body", recording_id="rec", workflow_kind="subject_mask_component"
        )
        runtime = SimpleNamespace(
            task_id="body",
            refined=SimpleNamespace(
                run_name="mask", group=SimpleNamespace(attrs={"edit_revision": 1})
            ),
        )
        for index in range(120):
            apply_id = f"apply-{index}"
            offer = dict(
                tail_refresh_status="complete",
                tail_refresh_version=f"version-{index}",
                tail_refresh_tasks=[],
                tail_refresh_failures=[],
                tail_refresh_valid_rows=2,
                tail_refresh_training_eligible_rows=1,
                tail_refresh_manual_point_count=4,
                tail_refresh_mask_revision=1,
            )
            store.record_event(
                task_id="body",
                recording_id="rec",
                user="reviewer",
                event_type=TAIL_SUCCESSOR_EVENT,
                target={"apply_id": apply_id},
                after={
                    "tail_refresh": offer,
                    "source_bindings": {
                        "mask_apply_refresh": {
                            "apply_id": apply_id,
                            "source_mask_run": "refined_subject_masks_runs/mask",
                            "source_mask_edit_revision": 1,
                        }
                    },
                },
            )
        assert (
            tail_successor_offer(
                store, runtime, apply_id="apply-0", expected_mask_revision=1
            )["tail_refresh_version"]
            == "version-0"
        )
        runtime.refined.group.attrs["edit_revision"] = 2
        assert tail_successor_offer(store, runtime) == {}
        with pytest.raises(RuntimeError, match="different mask revision"):
            tail_successor_offer(
                store, runtime, apply_id="apply-0", expected_mask_revision=2
            )
        runtime.refined.run_name = "different"
        with pytest.raises(RuntimeError, match="conflicting source binding"):
            tail_successor_offer(
                store, runtime, apply_id="apply-0", expected_mask_revision=1
            )
    finally:
        store.close()
