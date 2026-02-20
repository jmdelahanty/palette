from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.visualization.visualize_eye_mask_patches import (
    _apply_eye_mask_review_status,
    _append_flagged_frame,
    _build_eye_row,
    _extract_primary_reject_reason,
    _extract_patch_bounds,
    _fit_ellipse_from_mask,
    _format_reason_tags_compact,
    _load_frame_flags,
    _mouse_modifier_state,
    _reason_tags_from_value,
    _resolve_erase_mode,
    _save_roi_mask_edits,
    _step_clamped,
    parse_args,
    _resolve_eye_center,
    _resolve_eye_center_with_source,
)


def test_extract_patch_bounds_clamps_at_edges() -> None:
    x0, x1, y0, y1 = _extract_patch_bounds((20, 20), (1.2, 2.8), padding=5)
    assert (x0, x1, y0, y1) == (0, 7, 0, 9)


def test_resolve_eye_center_prefers_keypoint() -> None:
    keypoints = np.array(
        [
            [0.0, 0.0],   # bladder
            [10.0, 11.0], # left eye
            [20.0, 21.0], # right eye
        ],
        dtype=np.float32,
    )
    ellipse = np.array([5.0, 6.0, 8.0, 4.0, 10.0], dtype=np.float32)
    mask = np.zeros((12, 12), dtype=np.uint8)
    center = _resolve_eye_center(0, keypoints, ellipse, True, mask, (12, 12))
    assert center == (10.0, 11.0)


def test_resolve_eye_center_falls_back_to_ellipse_then_mask_centroid() -> None:
    keypoints_missing = np.full((3, 2), np.nan, dtype=np.float32)
    ellipse = np.array([7.5, 8.5, 8.0, 4.0, 0.0], dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2, 4] = 1
    mask[4, 6] = 1

    from_ellipse = _resolve_eye_center(1, keypoints_missing, ellipse, True, mask, (16, 16))
    assert from_ellipse == (7.5, 8.5)

    from_mask = _resolve_eye_center(
        1,
        keypoints_missing,
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
        False,
        mask,
        (16, 16),
    )
    assert from_mask[0] == 5.0
    assert from_mask[1] == 3.0


def test_resolve_eye_center_with_source_reports_resolution_path() -> None:
    keypoints_missing = np.full((3, 2), np.nan, dtype=np.float32)
    ellipse = np.array([7.5, 8.5, 8.0, 4.0, 0.0], dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2, 4] = 1
    mask[4, 6] = 1

    center_ellipse, source_ellipse = _resolve_eye_center_with_source(
        1,
        keypoints_missing,
        ellipse,
        True,
        mask,
        (16, 16),
    )
    assert source_ellipse == "ellipse"
    assert center_ellipse == (7.5, 8.5)

    center_mask, source_mask = _resolve_eye_center_with_source(
        1,
        keypoints_missing,
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
        False,
        mask,
        (16, 16),
    )
    assert source_mask == "mask_centroid"
    assert center_mask[0] == 5.0
    assert center_mask[1] == 3.0

    empty_mask = np.zeros((10, 14), dtype=np.uint8)
    center_roi, source_roi = _resolve_eye_center_with_source(
        1,
        keypoints_missing,
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
        False,
        empty_mask,
        (10, 14),
    )
    assert source_roi == "roi_center"
    assert center_roi == (7.0, 5.0)


def test_build_eye_row_includes_full_roi_panel() -> None:
    roi = np.zeros((12, 12), dtype=np.uint8)
    roi[4:8, 4:8] = 120
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[5:7, 5:7] = 1
    ellipse = np.array([6.0, 6.0, 6.0, 4.0, 20.0], dtype=np.float32)

    row = _build_eye_row(
        roi_img=roi,
        mask_eye=mask,
        ellipse_row=ellipse,
        ellipse_ok=True,
        center_xy=(6.0, 6.0),
        padding=2,
        eye_label="Left",
        eye_idx=0,
    )

    # Row contains 4 labeled panels: crop, mask, fit, and full ROI.
    assert row.shape[0] == 30
    assert row.shape[1] == 45


def test_append_flagged_frame_dedupes_and_sorts(tmp_path: Path) -> None:
    flag_path = tmp_path / "eye_mask_frame_flags.json"
    zarr_path = "/tmp/recording_training.zarr"

    _append_flagged_frame(flag_path, zarr_path, frame_idx=20, roi_idx=5)
    _append_flagged_frame(flag_path, zarr_path, frame_idx=10, roi_idx=None)
    _append_flagged_frame(flag_path, zarr_path, frame_idx=20, roi_idx=5)
    _append_flagged_frame(flag_path, zarr_path, frame_idx=20, roi_idx=1)

    payload = json.loads(flag_path.read_text(encoding="utf-8"))
    assert payload[zarr_path] == [
        {"frame_idx": 10, "roi_idx": None},
        {"frame_idx": 20, "roi_idx": 1},
        {"frame_idx": 20, "roi_idx": 5},
    ]


def test_append_flagged_frame_preserves_extra_metadata(tmp_path: Path) -> None:
    flag_path = tmp_path / "keypoint_nudge_flags.json"
    zarr_path = "/tmp/recording_training.zarr"

    _append_flagged_frame(
        flag_path,
        zarr_path,
        frame_idx=33,
        roi_idx=4,
        extra_fields={
            "action": "keypoint_nudge",
            "preserve_eye_masks": True,
            "requested_by": "reviewer_a",
        },
    )

    payload = json.loads(flag_path.read_text(encoding="utf-8"))
    assert payload[zarr_path] == [
        {
            "action": "keypoint_nudge",
            "frame_idx": 33,
            "preserve_eye_masks": True,
            "requested_by": "reviewer_a",
            "roi_idx": 4,
        }
    ]


def test_load_frame_flags_accepts_legacy_frame_lists(tmp_path: Path) -> None:
    flag_path = tmp_path / "legacy_frame_flags.json"
    flag_path.write_text(
        json.dumps(
            {
                "/tmp/a.zarr": [9, "bad", 1],
                "/tmp/b.zarr": [{"frame_idx": 5, "roi_idx": 2}, {"frame_idx": "x"}],
            }
        ),
        encoding="utf-8",
    )

    parsed = _load_frame_flags(flag_path)
    assert parsed["/tmp/a.zarr"] == [
        {"frame_idx": 9, "roi_idx": None},
        {"frame_idx": 1, "roi_idx": None},
    ]
    assert parsed["/tmp/b.zarr"] == [{"frame_idx": 5, "roi_idx": 2}]


def test_parse_args_sets_default_frame_flag_file() -> None:
    args = parse_args(["/tmp/example.zarr"])
    assert args.frame_flag_file == "eye_mask_frame_flags.json"
    assert args.keypoint_nudge_flag_file == "keypoint_nudge_flags.json"
    assert args.edit_zoom == 6
    assert args.review_state == "approved"
    assert args.review_method == "manual"
    assert args.review_intended_use == "training"


def test_apply_eye_mask_review_status_writes_contract_attrs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined = refined_parent.create_group("refined_eye_masks_001")
    refined.attrs.update(
        {
            "source_eye_masks_run": "eye_masks_001",
            "source_keypoints_run": "refined_keypoints_001",
            "source_keypoint_group": "refined_keypoints_runs",
        }
    )

    payload = _apply_eye_mask_review_status(
        refined_parent,
        "refined_eye_masks_001",
        refined,
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="reviewer_a",
        notes="looks good",
    )

    status = dict(refined.attrs["eye_mask_review_status"])
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"
    assert status["reviewer"] == "reviewer_a"
    assert status["notes"] == "looks good"
    assert status["source_eye_masks_run"] == "eye_masks_001"
    assert status["source_keypoints_run"] == "refined_keypoints_001"
    assert status["source_keypoint_group"] == "refined_keypoints_runs"
    assert "timestamp" in status
    assert payload["state"] == "approved"
    assert refined_parent.attrs["eye_mask_review_status_latest"] == "refined_eye_masks_001"


def test_mouse_modifier_state_decodes_ctrl_shift_lmb() -> None:
    cv2 = pytest.importorskip("cv2")
    flags = int(cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_SHIFTKEY | cv2.EVENT_FLAG_LBUTTON)
    ctrl, shift, lmb = _mouse_modifier_state(flags)
    assert ctrl is True
    assert shift is True
    assert lmb is True


def test_resolve_erase_mode_allows_shift_temporary_inverse() -> None:
    assert _resolve_erase_mode(False, False) is False
    assert _resolve_erase_mode(True, False) is True
    assert _resolve_erase_mode(False, True) is True
    assert _resolve_erase_mode(True, True) is False


def test_step_clamped_respects_bounds() -> None:
    assert _step_clamped(6, 1, 1, 40) == 7
    assert _step_clamped(1, -1, 1, 40) == 1
    assert _step_clamped(40, 5, 1, 40) == 40


def test_reason_helpers_extract_reject_tag_from_compound_reason() -> None:
    tags = _reason_tags_from_value("manual_correction|patch_viewer_edit|overlap")
    assert tags == ["manual_correction", "patch_viewer_edit", "overlap"]
    assert _extract_primary_reject_reason(tags) == "overlap"


def test_reason_helpers_compact_display_and_none_handling() -> None:
    assert _reason_tags_from_value(None) == []
    assert _extract_primary_reject_reason(["manual_correction", "retuned"]) is None
    compact = _format_reason_tags_compact(
        ["manual_correction", "patch_viewer_edit", "overlap", "retuned", "other"],
        max_items=3,
    )
    assert compact == "manual_correction|patch_viewer_edit|overlap|+2"


def test_fit_ellipse_from_mask_canonicalizes_major_minor() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.ellipse(mask, (32, 32), (18, 8), 25, 0, 360, 1, -1)
    params, success, contour, centroid = _fit_ellipse_from_mask(mask)
    assert success
    assert contour is not None
    assert centroid is not None
    assert np.isfinite(params).all()
    assert params[2] >= params[3]
    assert params[2] > 0
    assert params[3] > 0


def test_save_roi_mask_edits_updates_arrays(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined = refined_parent.create_group("refined_eye_masks_001")
    masks_arr = refined.create_array("masks_roi", shape=(1, 2, 32, 32), dtype="u1")
    ellipse_params = refined.create_array("ellipse_params", shape=(1, 2, 5), dtype="f4")
    ellipse_success = refined.create_array("ellipse_success", shape=(1, 2), dtype=bool)
    eye_sep = refined.create_array("eye_separation", shape=(1,), dtype="f4")

    edited = np.zeros((2, 32, 32), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.circle(edited[0], (10, 16), 5, 1, -1)
    cv2.circle(edited[1], (22, 16), 5, 1, -1)

    result = _save_roi_mask_edits(
        root=root,
        refined=refined,
        roi_idx=0,
        masks_row=edited,
        masks_arr=masks_arr,
        ellipse_params_arr=ellipse_params,
        ellipse_success_arr=ellipse_success,
        eye_separation_arr=eye_sep,
        reason_arr=None,
    )

    assert result["channel_count"] == 2
    assert result["successful_eyes"] == 2
    assert np.asarray(masks_arr[0]).sum() > 0
    assert bool(np.asarray(ellipse_success[0])[0])
    assert bool(np.asarray(ellipse_success[0])[1])
    assert float(np.asarray(eye_sep[0])) > 0.0
