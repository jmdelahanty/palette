"""Rotate tool: smoothed rotation of the active mask, with provenance on Save."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from fisheye.labeling import web
from fisheye.labeling.web_subject_mask_edit_operations import validated_edit_operations
from tests.unit.fisheye.test_labeling_web_routes import _running_server
from tests.unit.fisheye.test_mask_tail_apply_refresh import browser_context, reviewed_archive  # noqa: F401
from tests.unit.fisheye.test_subject_mask_editor_pieces import _as_mask, _run
from tests.unit.fisheye.test_web_mask_tail_apply_integration import request


def _ellipse(shape=(48, 48)):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return ((((xx - 23) / 15) ** 2 + ((yy - 24) / 6) ** 2) <= 1).astype(np.uint8)


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max(1, (a | b).sum())


def test_rotation_matches_opencv_bilinear_threshold():
    mask = _ellipse()
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    for degrees in (10, 30, 45, -20):
        pure = f"Array.from(rotateMask(mask, maskWidth, maskHeight, {degrees}, {cx}, {cy}))"
        ours = _as_mask(_run(mask, [], pure=pure)["pure"], mask.shape)
        # OpenCV's positive angle is counterclockwise; the editor's is clockwise on screen.
        M = cv2.getRotationMatrix2D((cx, cy), -degrees, 1.0)
        ref = (cv2.warpAffine(mask.astype(np.float32), M, mask.shape[::-1], flags=cv2.INTER_LINEAR) >= 0.5)
        assert _iou(ours, ref) > 0.98, degrees
        assert abs(int(ours.sum()) - int(ref.sum())) <= 3  # same area as OpenCV, within 3 px


def test_arrows_rotate_report_and_one_undo_reverts_and_drops_the_record():
    mask = _ellipse()
    out = _run(mask, [
        {"name": "tool", "press": {"key": "t"}},
        {"name": "five", "press": {"key": "ArrowRight", "mods": {"shiftKey": True}}},
        {"name": "six", "press": {"key": "ArrowRight"}},
        {"name": "end", "press": {"key": "b"}},
        {"name": "undo", "run": "undoBulkEdit()"},
    ], pure="JSON.stringify(pendingEditOperations)")
    assert "Rotated 6° clockwise" in out["six"]["status"] and "1 piece" in out["six"]["status"]
    assert not np.array_equal(_as_mask(out["six"]["mask"], mask.shape), mask)
    np.testing.assert_array_equal(_as_mask(out["undo"]["mask"], mask.shape), mask)
    assert json.loads(out["pure"]) == []


def test_finished_rotation_is_queued_for_save_with_its_angle():
    out = _run(_ellipse(), [
        {"name": "tool", "press": {"key": "t"}},
        {"name": "turn", "press": {"key": "ArrowLeft", "mods": {"shiftKey": True}}},
        {"name": "end", "press": {"key": "m"}},
    ], pure="JSON.stringify(pendingEditOperations)")
    assert json.loads(out["pure"]) == [
        {"op": "rotate", "angle_deg": -5, "method": "bilinear_threshold_0.5", "pivot": "mask_centroid"}
    ]


def test_a_rotation_that_splits_the_mask_says_so():
    mask = np.zeros((40, 40), np.uint8)
    mask[16:24, 4:14] = 1
    for i in range(18):  # a 1-px diagonal "tail"
        mask[19 - i // 3, 14 + i] = 1
    out = _run(mask, [{"name": "tool", "press": {"key": "t"}}, {"name": "turn", "run": "nudgeRotation(33)"}])
    assert "split the mask" in out["turn"]["status"] and "press R" in out["turn"]["status"]


@pytest.mark.parametrize("bad", [
    [], "rotate", [{"op": "rotate"}],
    [{"op": "move", "angle_deg": 5, "method": "bilinear_threshold_0.5", "pivot": "mask_centroid"}],
    [{"op": "rotate", "angle_deg": 5, "method": "nearest", "pivot": "mask_centroid"}],
    [{"op": "rotate", "angle_deg": float("nan"), "method": "bilinear_threshold_0.5", "pivot": "mask_centroid"}],
    [{"op": "rotate", "angle_deg": 0, "method": "bilinear_threshold_0.5", "pivot": "mask_centroid"}],
    [{"op": "rotate", "angle_deg": True, "method": "bilinear_threshold_0.5", "pivot": "mask_centroid"}],
    [{"op": "rotate", "angle_deg": 5, "method": "bilinear_threshold_0.5", "pivot": "mask_centroid", "extra": 1}],
])
def test_validator_refuses_anything_but_declared_rotations(bad):
    with pytest.raises(ValueError):
        validated_edit_operations(bad)


def test_validator_accepts_and_normalizes_rotations():
    ops = [{"op": "rotate", "angle_deg": 12.34567, "method": "bilinear_threshold_0.5", "pivot": "mask_centroid"}]
    assert validated_edit_operations(ops) == [{**ops[0], "angle_deg": 12.346}]
    assert validated_edit_operations(None) is None


def test_save_route_records_rotations_and_refuses_malformed_ones(reviewed_archive, tmp_path):
    path, root, initial = reviewed_archive
    store, _runtime = browser_context(reviewed_archive, tmp_path)
    store.upsert_labeling_user(user_id="reviewer", status="active")
    session = store.create_session(task_id="original-mask", user="reviewer")
    route = f"/api/sessions/{session.session_id}/subject-mask"
    mask = (np.asarray(root[initial["paths"]["mask_edit"]]["masks_roi"][1, 0]) > 0).astype(np.uint8)
    good = [{"op": "rotate", "angle_deg": 7.5, "method": "bilinear_threshold_0.5", "pivot": "mask_centroid"}]
    try:
        with _running_server(store, user="reviewer") as base:
            status, nav = request(base, route + "/nav", {"roi_idx": 1})
            status, refused = request(base, route + "/save", {
                "mask": web._raw_array_payload(mask), "target_token": nav["state"]["target_token"],
                "advance": False, "edit_operations": [{"op": "rotate", "angle_deg": 5}],
            })
            assert status == 400 and refused["error"] == "save_error"
            assert store.get_session_checkpoint(task_id="original-mask", roi_idx=1, component_name="subject_body") is None
            status, saved = request(base, route + "/save", {
                "mask": web._raw_array_payload(mask), "target_token": nav["state"]["target_token"],
                "advance": False, "edit_operations": good,
            })
            assert status == 200, saved
        checkpoint = store.get_session_checkpoint(task_id="original-mask", roi_idx=1, component_name="subject_body")
        assert checkpoint["metadata"]["edit_operations"] == good
    finally:
        store.close()
