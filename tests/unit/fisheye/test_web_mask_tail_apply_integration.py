"""Real HTTP mask Apply, version publication, and durable successor offers.

Prepared before combining the independently tested QC and tail candidates.
"""

import json
import urllib.error
import urllib.request

import numpy as np
import pytest
import zarr

from fisheye.labeling import web
from fisheye.training.recovered_mask_review_payload import array_hashes
from fisheye.tune.refined_subject_mask_review import _refined_subject_write_lock
from tests.unit.fisheye.test_labeling_web_routes import _running_server
from tests.unit.fisheye.test_mask_tail_apply_refresh import (
    browser_context,
    reviewed_archive,
)


def request(base, route, payload=None):
    req = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        response = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, json.loads(response.read())


@pytest.fixture
def context(reviewed_archive, tmp_path):
    store, runtime = browser_context(reviewed_archive, tmp_path)
    store.upsert_labeling_user(user_id="reviewer", status="active")
    pose_session = store.create_session(task_id="original-pose", user="reviewer")
    mask_session = store.create_session(task_id="original-mask", user="reviewer")
    try:
        yield store, runtime, pose_session, mask_session
    finally:
        store.close()


def save_mask(base, session_id, mask, *, position=1):
    route = f"/api/sessions/{session_id}/subject-mask"
    status, nav = request(base, route + "/nav", {"position": position})
    assert status == 200, nav
    status, saved = request(
        base,
        route + "/save",
        {
            "mask": web._raw_array_payload(mask),
            "target_token": nav["state"]["target_token"],
        },
    )
    assert status == 200, saved
    return saved["state"]["target_token"]


@pytest.mark.parametrize("repair", [True, False], ids=["repaired", "still_fragmented"])
def test_http_apply_offers_durable_successor_and_keeps_original_session(
    reviewed_archive, context, repair
):
    path, root, initial = reviewed_archive
    store, runtime, pose_session, mask_session = context
    pose = root[initial["paths"]["pose_edit"]]
    original_points = np.asarray(pose["keypoints_roi"][:])
    manual = np.asarray(pose["keypoint_manual_edit"][:], dtype=bool)
    original_pose_hashes = array_hashes(pose)
    original_task = store.get_task("original-pose")
    mask = root[initial["paths"]["mask_edit"]]
    changed = np.asarray(mask["masks_roi"][0 if repair else 1, 0])
    route = f"/api/sessions/{mask_session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        token = save_mask(base, mask_session.session_id, changed)
        status, applied = request(
            base, route + "/apply", {"apply_id": "refresh", "target_token": token}
        )
        assert status == 200, applied
        result = applied["result"]
        assert result["qc_status"] == "complete"
        assert applied["state"]["qc_status"] == "complete"
        assert result["tail_refresh_status"] == "complete"
        assert result["tail_refresh_valid_rows"] == (2 if repair else 1)
        assert result["tail_refresh_training_eligible_rows"] == (2 if repair else 1)
        failures = result["tail_refresh_failures"]
        assert [row["roi_idx"] for row in failures] == ([] if repair else [1])
        if failures:
            assert failures[0]["source_frame_idx"] == 31
            assert failures[0]["reason"]
        assert store.get_task("original-pose") == original_task
        assert store.get_session(pose_session.session_id)["closed_at_utc"] is None

        task = next(
            t for t in result["tail_refresh_tasks"] if t["workflow_kind"] == "keypoints"
        )
        status, opened = request(
            base, f"/api/tasks/{task['task_id']}/open", {"expected_user": "reviewer"}
        )
        assert status == 200, opened
        assert opened["session"]["task_id"] == task["task_id"]
        assert store.get_session(pose_session.session_id)["closed_at_utc"] is None
        new_run = store.get_task(task["task_id"])["run_name"]
        current = zarr.open_group(str(path), mode="r", use_consolidated=False)
        np.testing.assert_array_equal(
            current[f"refined_keypoints_runs/{new_run}"]["keypoints_roi"][:][manual],
            original_points[manual],
        )
        assert (
            array_hashes(current[initial["paths"]["pose_edit"]]) == original_pose_hashes
        )

    # Reopening the server/browser and retrying an already-complete Apply must
    # recover the same offer even if the first success response was lost.
    with _running_server(store, user="reviewer") as base:
        status, state = request(base, route + "/state")
        assert status == 200, state
        assert (
            state["state"]["tail_refresh"]["tail_refresh_version"]
            == result["tail_refresh_version"]
        )
        status, replay = request(
            base,
            route + "/apply",
            {
                "apply_id": "refresh",
                "target_token": state["state"]["target_token"],
            },
        )
        assert status == 200, replay
        assert (
            replay["result"]["tail_refresh_version"] == result["tail_refresh_version"]
        )
        assert (
            store.count_pending_session_checkpoint_apply_effects(
                task_id="original-mask"
            )
            == 0
        )


def test_tail_publication_failure_blocks_sibling_task_until_locked_retry(
    reviewed_archive, context, monkeypatch
):
    from fisheye.training import mask_tail_apply_refresh as publisher

    path, root, initial = reviewed_archive
    store, runtime, pose_session, mask_session = context
    store.upsert_task(
        task_id="sibling-swim",
        recording_id="rec",
        workflow_kind="subject_mask_component",
        component_name="swim_bladder",
        run_name=runtime.refined.run_name,
        scope={"zarr_path": str(path), "refined_run": runtime.refined.run_name},
    )
    sibling = store.create_session(task_id="sibling-swim", user="reviewer")
    original_publish = publisher.publish_review_payload
    failed = False

    def publish_then_fail(*args, **kwargs):
        nonlocal failed
        result = original_publish(*args, **kwargs)
        if not failed:
            failed = True
            raise OSError("injected after tail publication")
        return result

    monkeypatch.setattr(publisher, "publish_review_payload", publish_then_fail)
    route = f"/api/sessions/{mask_session.session_id}/subject-mask"
    sibling_route = f"/api/sessions/{sibling.session_id}/subject-mask"
    mask = root[initial["paths"]["mask_edit"]]
    with _running_server(store, user="reviewer") as base:
        body_token = save_mask(
            base, mask_session.session_id, np.asarray(mask["masks_roi"][0, 0])
        )
        status, failed_response = request(
            base,
            route + "/apply",
            {"apply_id": "retry-tail", "target_token": body_token},
        )
        assert status == 400, failed_response
        assert failed_response["error"] == "subject_mask_apply_effects_pending"
        assert failed_response["canonical_apply_succeeded"] is True
        assert (
            store.count_pending_session_checkpoint_apply_effects(
                task_id="original-mask"
            )
            == 1
        )
        before = zarr.open_group(str(path), mode="r", use_consolidated=False)[
            initial["paths"]["mask_edit"]
        ]
        pixels = np.asarray(before["masks_roi"][:])
        revisions = np.asarray(before["components/subject_body/row_revision"][:])
        edit_revision = before.attrs["edit_revision"]
        status, sibling_state = request(base, sibling_route + "/state")
        assert status == 200, sibling_state
        sibling_token = sibling_state["state"]["target_token"]
        status, blocked = request(
            base,
            sibling_route + "/review-status",
            {"state": "approved", "target_token": sibling_token},
        )
        assert status == 409, blocked
        assert blocked["error"] == "pending_apply_effects"
        status, blocked = request(
            base,
            "/api/tasks/sibling-swim/complete",
            {"session_id": sibling.session_id, "expected_user": "reviewer"},
        )
        assert status == 409, blocked
        assert blocked["error"] == "pending_apply_effects"

        sibling_token = save_mask(
            base, sibling.session_id, np.asarray(mask["masks_roi"][1, 2])
        )
        status, blocked = request(
            base,
            sibling_route + "/apply",
            {"apply_id": "sibling", "target_token": sibling_token},
        )
        assert status == 400, blocked

    original_complete = store.mark_session_checkpoint_apply_effects_complete
    checked_lock = []

    def complete_under_lock(**kwargs):
        with pytest.raises(TimeoutError):
            with _refined_subject_write_lock(
                path, refined_run=runtime.refined.run_name, timeout_seconds=0.01
            ):
                pass
        checked_lock.append(True)
        return original_complete(**kwargs)

    monkeypatch.setattr(
        store, "mark_session_checkpoint_apply_effects_complete", complete_under_lock
    )
    with _running_server(store, user="reviewer") as base:
        status, state = request(base, route + "/state")
        assert status == 200, state
        status, retried = request(
            base,
            route + "/apply",
            {"apply_id": "retry-tail", "target_token": state["state"]["target_token"]},
        )
        assert status == 200, retried
        assert retried["result"]["already_applied"] is True
        assert retried["result"]["tail_refresh_status"] == "complete"
        assert checked_lock
        after = zarr.open_group(str(path), mode="r", use_consolidated=False)[
            initial["paths"]["mask_edit"]
        ]
        np.testing.assert_array_equal(after["masks_roi"][:], pixels)
        np.testing.assert_array_equal(
            after["components/subject_body/row_revision"][:], revisions
        )
        assert after.attrs["edit_revision"] == edit_revision
        assert (
            store.count_pending_session_checkpoint_apply_effects(
                task_id="original-mask"
            )
            == 0
        )
        assert (
            store.count_unapplied_session_checkpoints(
                task_id="sibling-swim", component_name="swim_bladder"
            )
            == 1
        )
        status, sibling_state = request(base, sibling_route + "/state")
        assert status == 200, sibling_state
        assert sibling_state["state"]["pending_apply_effect_count"] == 0
        assert store.get_session(pose_session.session_id)["closed_at_utc"] is None


def test_registry_retry_reuses_completed_snapshot_and_retains_later_source_labels(
    reviewed_archive, context, monkeypatch
):
    from fisheye.tune import keypoint_review_backend as editor

    path, root, initial = reviewed_archive
    store, runtime, pose_session, mask_session = context
    registry_calls = []

    def registry_effect(**kwargs):
        registry_calls.append(kwargs)
        return len(registry_calls) > 1

    monkeypatch.setattr(web, "_refresh_registry_for_scope", registry_effect)
    route = f"/api/sessions/{mask_session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        mask = root[initial["paths"]["mask_edit"]]
        token = save_mask(
            base, mask_session.session_id, np.asarray(mask["masks_roi"][0, 0])
        )
        status, failed = request(
            base,
            route + "/apply",
            {"apply_id": "registry-retry", "target_token": token},
        )
        assert status == 400, failed
        assert "registry" in failed["details"]
        event = store.get_event_for_target(
            task_id="original-mask",
            event_type="mask_apply_tail_successor",
            target={"apply_id": "registry-retry"},
        )
        version = event["after"]["version"]
        successor_id = next(
            t["task_id"]
            for t in event["after"]["tasks"]
            if t["workflow_kind"] == "keypoints"
        )
        store.update_task_state(task_id=successor_id, state="complete")
        source = editor.resolve_review_session(
            str(path),
            refined_run=initial["paths"]["pose_edit"].split("/")[1],
            include_all=True,
        )
        later_points = np.asarray(source.kp_roi_arr[0]).copy()
        later_points[14, 0] += 2
        editor.save_roi_correction(source, position=0, points=later_points)
        # This later checkpoint also belongs to the unchanged original task. A
        # retry of completed publication must neither consume it nor rebase it.
        store.upsert_session_checkpoint(
            session_id=pose_session.session_id,
            task_id="original-pose",
            recording_id="rec",
            user="reviewer",
            workflow_kind="keypoints",
            target_run_path=initial["paths"]["pose_edit"],
            target_edit_revision=0,
            source_rowset_path=initial["paths"]["crop"],
            roi_idx=0,
            component_name="keypoints",
            payload={"later_checkpoint": True},
            metadata={},
        )

    with _running_server(store, user="reviewer") as base:
        status, state = request(base, route + "/state")
        assert status == 200, state
        status, retried = request(
            base,
            route + "/apply",
            {
                "apply_id": "registry-retry",
                "target_token": state["state"]["target_token"],
            },
        )
        assert status == 200, retried
        assert retried["result"]["tail_refresh_version"] == version
        assert store.get_task(successor_id)["state"] == "complete"
        np.testing.assert_array_equal(source.kp_roi_arr[0], later_points)
        assert (
            store.count_unapplied_session_checkpoints(
                task_id="original-pose", component_name="keypoints"
            )
            == 1
        )
        assert (
            store.count_pending_session_checkpoint_apply_effects(
                task_id="original-mask"
            )
            == 0
        )
        status, approved = request(
            base,
            route + "/review-status",
            {"state": "approved", "target_token": state["state"]["target_token"]},
        )
        assert status == 200, approved
        status, completed = request(
            base,
            "/api/tasks/original-mask/complete",
            {"session_id": mask_session.session_id, "expected_user": "reviewer"},
        )
        assert status == 200, completed
        assert store.get_session(pose_session.session_id)["closed_at_utc"] is None
