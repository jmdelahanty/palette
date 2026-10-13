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
from fisheye.training.mask_tail_border_acceptance import ATTR as TAIL_ACCEPTANCE_ATTR
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


def test_http_accept_and_revoke_visible_endpoint_without_painting(reviewed_archive, context):
    path, root, initial = reviewed_archive
    store, runtime, pose_session, mask_session = context
    mask_run = root[initial["paths"]["mask_edit"]]
    body = np.asarray(mask_run["masks_roi"][0, 0]).copy()
    body[108:128, 64] = 1
    mask_run["masks_roi"][0, 0] = body
    before = np.asarray(mask_run["masks_roi"][:]).copy()
    route = f"/api/sessions/{mask_session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        status, current = request(base, route + "/roi/current")
        assert status == 200, current
        assert "body_touches_crop_border" in current["tail_crop_border"]["original_queued_reason"] or current["tail_crop_border"]["original_queued_reason"]
        status, saved = request(base, route + "/save", {
            "mask": web._raw_array_payload(body),
            "tail_crop_border_action": {"action": "accept", "reason": "Only the tiny visible tail tip is clipped", "accepted_by": "forged-user", "accepted_at_utc": "2000-01-01T00:00:00Z"},
            "target_token": current["state"]["target_token"],
        })
        assert status == 200, saved
        status, checkpoint_status = request(base, route + "/roi/status")
        assert status == 200, checkpoint_status
        assert checkpoint_status["tail_crop_border"]["pending_action"]["action"] == "accept"
        status, applied = request(base, route + "/apply", {
            "apply_id": "accept-tail", "target_token": saved["state"]["target_token"],
        })
        assert status == 200, applied
        assert applied["result"]["tail_refresh_visible_endpoint_rows"] == [0]
        status, current_status = request(base, route + "/roi/status")
        assert status == 200, current_status
        assert current_status["tail_crop_border"]["accepted"] is True
        assert current_status["tail_crop_border"]["latest_outcome"]["visible_endpoint"] is True
        np.testing.assert_array_equal(mask_run["masks_roi"][:], before)
        fresh_mask = zarr.open_group(str(path), mode="r", use_consolidated=False)[initial["paths"]["mask_edit"]]
        record = fresh_mask.attrs[TAIL_ACCEPTANCE_ATTR]["0"]
        assert record["accepted_by"] == "reviewer"
        assert record["accepted_at_utc"] != "2000-01-01T00:00:00Z"
        assert record["reason"] == "Only the tiny visible tail tip is clipped"

        status, state = request(base, route + "/state")
        status, old_client_saved = request(base, route + "/save", {
            "mask": web._raw_array_payload(body),
            "target_token": state["state"]["target_token"],
        })
        assert status == 200, old_client_saved
        status, old_client_apply = request(base, route + "/apply", {
            "apply_id": "unchanged-old-client", "target_token": old_client_saved["state"]["target_token"],
        })
        assert status == 200, old_client_apply
        assert old_client_apply["result"]["tail_refresh_visible_endpoint_rows"] == [0]
        fresh_mask = zarr.open_group(str(path), mode="r", use_consolidated=False)[initial["paths"]["mask_edit"]]
        assert fresh_mask.attrs[TAIL_ACCEPTANCE_ATTR]["0"]["accepted_at_mask_revision"] == record["accepted_at_mask_revision"]

        status, state = request(base, route + "/state")
        status, saved = request(base, route + "/save", {
            "mask": web._raw_array_payload(body),
            "tail_crop_border_action": {"action": "revoke"},
            "target_token": state["state"]["target_token"],
        })
        assert status == 200, saved
        status, revoked = request(base, route + "/apply", {
            "apply_id": "revoke-tail", "target_token": saved["state"]["target_token"],
        })
        assert status == 200, revoked
        assert 0 not in revoked["result"].get("tail_refresh_visible_endpoint_rows", [])
        assert any(f["roi_idx"] == 0 and f["reason"] == "body_touches_crop_border" for f in revoked["result"]["tail_refresh_failures"])
        fresh_mask = zarr.open_group(str(path), mode="r", use_consolidated=False)[initial["paths"]["mask_edit"]]
        assert "0" not in fresh_mask.attrs[TAIL_ACCEPTANCE_ATTR]
        np.testing.assert_array_equal(mask_run["masks_roi"][:], before)


def test_http_changed_body_after_accept_checkpoint_clears_pending_action(reviewed_archive, context):
    path, root, initial = reviewed_archive
    store, runtime, pose_session, mask_session = context
    body = np.asarray(root[initial["paths"]["mask_edit"]]["masks_roi"][0, 0]).copy()
    body[108:128, 64] = 1
    changed = body.copy()
    changed[65, 64] = 0
    route = f"/api/sessions/{mask_session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        status, current = request(base, route + "/roi/current")
        assert status == 200, current
        status, accepted_checkpoint = request(base, route + "/save", {
            "mask": web._raw_array_payload(body),
            "tail_crop_border_action": {"action": "accept", "reason": "Slight tip clipping is acceptable"},
            "target_token": current["state"]["target_token"],
        })
        assert status == 200, accepted_checkpoint
        status, later_checkpoint = request(base, route + "/save", {
            "mask": web._raw_array_payload(changed),
            "target_token": accepted_checkpoint["state"]["target_token"],
        })
        assert status == 200, later_checkpoint
        status, row_status = request(base, route + "/roi/status")
        assert status == 200, row_status
        assert row_status["tail_crop_border"]["pending_action"] is None
        status, applied = request(base, route + "/apply", {
            "apply_id": "changed-after-accept", "target_token": later_checkpoint["state"]["target_token"],
        })
        assert status == 200, applied
        assert 0 not in applied["result"].get("tail_refresh_visible_endpoint_rows", [])
        assert any(f["roi_idx"] == 0 for f in applied["result"]["tail_refresh_failures"])
        fresh = zarr.open_group(str(path), mode="r", use_consolidated=False)[initial["paths"]["mask_edit"]]
        assert TAIL_ACCEPTANCE_ATTR not in fresh.attrs


@pytest.mark.parametrize("repair", [True, False], ids=["repaired", "still_fragmented"])
def test_http_apply_offers_durable_successor_and_supersedes_original_pose(
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
        assert TAIL_ACCEPTANCE_ATTR not in zarr.open_group(str(path), mode="r", use_consolidated=False)[initial["paths"]["mask_edit"]].attrs
        assert result["tail_refresh_valid_rows"] == (2 if repair else 1)
        assert result["tail_refresh_training_eligible_rows"] == (2 if repair else 1)
        failures = result["tail_refresh_failures"]
        assert [row["roi_idx"] for row in failures] == ([] if repair else [1])
        if failures:
            assert failures[0]["source_frame_idx"] == 31
            assert failures[0]["reason"]
        # Enforcement correction (2026-09-26): the replaced pose task is
        # superseded and its session closed, so later edits cannot be stranded.
        superseded = store.get_task("original-pose")
        assert superseded["state"] == "superseded"
        assert {k: v for k, v in superseded.items() if k not in {"state", "updated_at_utc"}} == {
            k: v for k, v in original_task.items() if k not in {"state", "updated_at_utc"}
        }
        assert store.get_session(pose_session.session_id)["closed_at_utc"] is not None
        assert store.get_task("original-mask")["state"] == "pending"

        task = next(
            t for t in result["tail_refresh_tasks"] if t["workflow_kind"] == "keypoints"
        )
        status, opened = request(
            base, f"/api/tasks/{task['task_id']}/open", {"expected_user": "reviewer"}
        )
        assert status == 200, opened
        assert opened["session"]["task_id"] == task["task_id"]
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
        clipped = np.asarray(mask["masks_roi"][0, 0]).copy()
        clipped[108:128, 64] = 1
        status, current = request(base, route + "/roi/current")
        assert status == 200, current
        status, saved = request(base, route + "/save", {
            "mask": web._raw_array_payload(clipped),
            "tail_crop_border_action": {"action": "accept", "reason": "Tiny visible tail endpoint is usable"},
            "target_token": current["state"]["target_token"],
        })
        assert status == 200, saved
        body_token = saved["state"]["target_token"]
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
        assert retried["result"]["tail_refresh_visible_endpoint_rows"] == [0]
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
        # Another component's task on the applied run stays open; the replaced
        # pose task is superseded once the successor is published.
        assert store.get_task("sibling-swim")["state"] == "pending"
        assert store.get_task("original-pose")["state"] == "superseded"
        assert store.get_session(pose_session.session_id)["closed_at_utc"] is not None


def test_registry_retry_reuses_completed_snapshot_and_refuses_later_source_edits(
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
        # The original pose task was superseded when the successor published,
        # so a later browser checkpoint there is refused instead of stranded.
        # (The direct Zarr edit above stands in for historical data.) A retry
        # of the completed publication must not rebase onto that edit.
        with pytest.raises(RuntimeError, match="replaced by a newer one"):
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
            == 0
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
        assert store.get_session(pose_session.session_id)["closed_at_utc"] is not None
