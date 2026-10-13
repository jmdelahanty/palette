"""Mask Apply staleness is per component content, not the run-wide revision.

A refined mask run has one edit_revision for all components. Applying one
component used to make every other component's saved rows stale; now a save
whose own component row is unchanged since it was made still applies.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.labeling import web
from fisheye.labeling.web_subject_mask_apply_state import component_row_sha256
from tests.unit.fisheye.test_labeling_web_routes import _running_server
from tests.unit.fisheye.test_mask_tail_apply_refresh import browser_context, reviewed_archive  # noqa: F401
from tests.unit.fisheye.test_web_mask_tail_apply_integration import request


@pytest.fixture
def two_components(reviewed_archive, tmp_path):
    path, root, initial = reviewed_archive
    store, _runtime = browser_context(reviewed_archive, tmp_path)
    store.upsert_labeling_user(user_id="reviewer", status="active")
    task = store.get_task("original-mask")
    store.upsert_task(recording_id="rec", task_id="eyes-task", workflow_kind="subject_mask_component",
                      run_name=task["run_name"], component_name="eyes_union",
                      scope={**task["scope"], "component_name": "eyes_union"})
    run_rel = initial["paths"]["mask_edit"]
    run = root[run_rel]
    body = np.asarray(run["masks_roi"][1, 0]).copy()
    body[4:9, 100:107] = 1
    eyes = np.asarray(run["masks_roi"][1, 1]).copy()
    eyes[110:114, 10:14] = 1
    try:
        yield store, path, run_rel, body, eyes
    finally:
        store.close()


def _session_save(store, task_id, roi, mask):
    session = store.create_session(task_id=task_id, user="reviewer")
    route = f"/api/sessions/{session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        status, nav = request(base, route + "/nav", {"roi_idx": roi})
        assert status == 200, nav
        status, saved = request(base, route + "/save", {
            "mask": web._raw_array_payload(mask), "target_token": nav["state"]["target_token"], "advance": False,
        })
        assert status == 200, saved
    store.close_session(session_id=session.session_id, user="reviewer")


def _apply(store, task_id, apply_id):
    session = store.create_session(task_id=task_id, user="reviewer")
    route = f"/api/sessions/{session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        status, state = request(base, route + "/state")
        assert status == 200, state
        status, applied = request(base, route + "/apply", {"apply_id": apply_id, "target_token": state["state"]["target_token"]})
    store.close_session(session_id=session.session_id, user="reviewer")
    assert status == 200, applied
    return applied["result"]


def _run(path, run_rel):
    return zarr.open_group(str(path), mode="r", use_consolidated=False)[run_rel]


def test_saves_record_their_component_base_digest(two_components):
    store, path, run_rel, body, _eyes = two_components
    base = np.asarray(_run(path, run_rel)["masks_roi"][1, 0])
    _session_save(store, "original-mask", 1, body)
    checkpoint = store.get_session_checkpoint(task_id="original-mask", roi_idx=1, component_name="subject_body")
    assert checkpoint["metadata"]["base_component_sha256"] == component_row_sha256(base)


def test_two_components_saved_together_apply_in_sequence(two_components):
    store, path, run_rel, body, eyes = two_components
    _session_save(store, "original-mask", 1, body)
    _session_save(store, "eyes-task", 1, eyes)
    revision = _run(path, run_rel).attrs["edit_revision"]

    first = _apply(store, "eyes-task", "eyes-apply")
    assert first["applied_checkpoint_count"] == 1
    assert _run(path, run_rel).attrs["edit_revision"] == revision + 1

    second = _apply(store, "original-mask", "body-apply")
    assert second["applied_checkpoint_count"] == 1, second
    assert int(second.get("stale_checkpoint_count") or 0) == 0
    run = _run(path, run_rel)
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][1, 0]), (body > 0).astype(np.uint8))
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][1, 1]), (eyes > 0).astype(np.uint8))


def test_a_save_whose_own_component_changed_is_still_stale(two_components):
    store, path, run_rel, body, eyes = two_components
    _session_save(store, "original-mask", 1, body)
    _session_save(store, "eyes-task", 1, eyes)
    _apply(store, "eyes-task", "eyes-apply")
    run = zarr.open_group(str(path), mode="a", use_consolidated=False)[run_rel]
    changed = np.asarray(run["masks_roi"][1, 0]).copy()
    changed[0:3, 0:3] = 1 - changed[0:3, 0:3]
    run["masks_roi"][1, 0] = changed
    result = _apply(store, "original-mask", "body-apply")
    assert result["applied_checkpoint_count"] == 0 and result["stale_checkpoint_count"] == 1
    np.testing.assert_array_equal(np.asarray(_run(path, run_rel)["masks_roi"][1, 0]), changed)


def test_saves_without_a_recorded_base_keep_the_revision_rule(two_components):
    store, path, run_rel, body, eyes = two_components
    _session_save(store, "eyes-task", 1, eyes)
    revision = _run(path, run_rel).attrs["edit_revision"]
    legacy_session = store.create_session(task_id="original-mask", user="reviewer")
    store.upsert_session_checkpoint(
        session_id=legacy_session.session_id, task_id="original-mask", recording_id="rec", user="reviewer",
        workflow_kind="subject_mask_component", target_run_path=run_rel, target_edit_revision=revision,
        source_rowset_path=None, roi_idx=1, component_name="subject_body",
        payload={"schema": "palette.web_labeling_subject_mask_checkpoint_payload.v1",
                 "payload_kind": "dense_roi_replacement_mask", "mask": web._raw_array_payload((body > 0).astype(np.uint8))},
        metadata={"schema": "palette.web_labeling_subject_mask_checkpoint_metadata.v1", "component_name": "subject_body",
                  "target_run_path": run_rel},
    )
    store.close_session(session_id=legacy_session.session_id, user="reviewer")
    _apply(store, "eyes-task", "eyes-apply")
    result = _apply(store, "original-mask", "body-apply")
    assert result["applied_checkpoint_count"] == 0 and result["stale_checkpoint_count"] == 1
