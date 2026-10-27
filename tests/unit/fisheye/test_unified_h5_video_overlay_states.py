"""Video tracking overlay states: paths the revision 7 witnesses do not show.

These tests edit scratch copies of the Citrus complete witness and call the
states validator directly (integrity would refuse any edit first). They show
that Palette is consistent with its own reading of the contract, not that it
agrees with Citrus: only the witnesses (and their C++ production rows) do
that. Drawn rows, the rig homography and the probe horizon are witnessed and
tested in ``test_unified_h5_admission_v7``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.correspondence import PREFLIGHT
from fisheye.shared.unified_h5.schema import admitted_binding
from fisheye.shared.unified_h5.video_overlay import (
    SETTINGS,
    STATES,
    box_arena_local,
    quantise_rgba8,
    validate_video_overlay_states,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "citrus_unified_h5_video_overlay_v1"
COMPLETE = FIXTURES / "video_overlay_v1_complete.h5"
BOXES = "/observations/bounding_boxes"


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "overlay.h5"
    shutil.copyfile(COMPLETE, target)
    target.chmod(0o644)
    return target


def _validate(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        document = json.loads(h5[PREFLIGHT][()])
        binding = admitted_binding(h5, document["admission_policy_sha256"], document)
        return validate_video_overlay_states(h5, binding, camera_serial="CAM-42")


def _edit_settings(path: Path, change) -> None:
    with h5py.File(path, "r+") as h5:
        settings = json.loads(h5[SETTINGS][()])
        change(settings)
        del h5[SETTINGS]
        h5.create_dataset(SETTINGS, data=json.dumps(settings), dtype=h5py.string_dtype())


def _edit_rows(path: Path, dataset: str, change) -> None:
    with h5py.File(path, "r+") as h5:
        rows = h5[dataset][()]
        change(rows)
        h5[dataset][...] = rows


def test_the_unedited_copy_is_admitted(tmp_path):
    assert _validate(_copy(tmp_path)) == {"rows": 4, "superseded_box_rows": 0}


def test_held_policy_skip_with_undrawn_held_rows_is_admitted(tmp_path):
    path = _copy(tmp_path)
    _edit_settings(path, lambda s: s.update(held_policy="skip"))

    def skip_held(rows):
        rows["decision_draw"][rows["input_used"] == 2] = 0

    _edit_rows(path, STATES, skip_held)
    assert _validate(path)["rows"] == 4


def test_held_policy_skip_with_a_held_decision_is_refused(tmp_path):
    path = _copy(tmp_path)
    _edit_settings(path, lambda s: s.update(held_policy="skip"))
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_held_policy_skip"):
        _validate(path)


def test_a_later_bounding_box_update_is_accepted_as_superseded(tmp_path):
    path = _copy(tmp_path)

    def later(rows):
        rows["payload_timestamp_ns_monotonic"][rows["recording_frame_id"] == 1] = 6000

    _edit_rows(path, BOXES, later)
    assert _validate(path) == {"rows": 4, "superseded_box_rows": 4}


def test_an_earlier_bounding_box_update_is_refused(tmp_path):
    path = _copy(tmp_path)

    def earlier(rows):
        rows["payload_timestamp_ns_monotonic"][rows["recording_frame_id"] == 1] = 4000

    _edit_rows(path, BOXES, earlier)
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_box_row_earlier"):
        _validate(path)


def test_a_settings_scale_other_than_the_calibration_snapshot_is_refused(tmp_path):
    path = _copy(tmp_path)
    _edit_settings(path, lambda s: s.update(pixels_per_mm_projector=10.0, radius_px=10.0))
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_scale_authority_mismatch"):
        _validate(path)


def test_a_radius_not_derived_from_the_scale_is_refused(tmp_path):
    path = _copy(tmp_path)
    _edit_settings(path, lambda s: s.update(radius_px=13.0))
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_settings_radius_px"):
        _validate(path)


def test_an_active_arrow_without_pose_enabled_is_refused(tmp_path):
    path = _copy(tmp_path)
    _edit_settings(path, lambda s: s.update(pose_observation_enabled=False))
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_pose_presence_mismatch"):
        _validate(path)


def test_a_misquantised_colour_is_refused(tmp_path):
    path = _copy(tmp_path)
    _edit_settings(path, lambda s: s.update(color_quantised_rgba=[0, 254, 0, 255]))
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_settings_colour_quantised:dot"):
        _validate(path)


def test_a_held_row_with_a_heading_is_refused(tmp_path):
    path = _copy(tmp_path)

    def heading_on_held(rows):
        rows["heading_status"][1] = 5

    _edit_rows(path, STATES, heading_on_held)
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_held_heading"):
        _validate(path)


def test_a_source_camera_other_than_the_binding_is_refused(tmp_path):
    path = _copy(tmp_path)
    _edit_rows(path, STATES, lambda rows: rows["source_camera_id"].__setitem__(slice(None), 9))
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_source_camera_mismatch"):
        _validate(path)


def test_quantiser_rounds_half_up_from_the_binary32_value():
    assert quantise_rgba8([0.0, 1.0, 0.5, 1.0]) == [0, 255, 128, 255]
    assert quantise_rgba8([float("nan"), -0.5, 2.0, 0.2]) == [0, 0, 255, 51]


def test_bounding_rect_floors_min_and_max_like_opencv():
    # Independent of the witnesses: x corners 0.25..10.75 floor to 0 and 10,
    # so the integer rectangle is x 0, width 11, centre 5.5.
    identity = np.eye(3)
    ok, cx, cy = box_arena_local(identity, (0, 0), [0.25], [0.25], [10.5], [10.5])
    assert ok.tolist() == [True]
    assert (float(cx[0]), float(cy[0])) == (5.5, 5.5)
    # A corner on the w = 0 horizon has no position.
    horizon = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.0]])
    ok, _, _ = box_arena_local(horizon, (0, 0), [0.0], [0.0], [1.0], [1.0])
    assert ok.tolist() == [False]


DEFECT_BUILD = "2228a0b1c1fac0d42df1e1d68e7bb0bb06ac3a93"
CANVAS = "/geometry/runtime/canvas_arena_config_json"


def _canvas_file_scale(path: Path, value: float) -> None:
    """The defective writer's shape: settings carry the canvas file's own scale."""

    with h5py.File(path, "r+") as h5:
        config = json.loads(h5[CANVAS][()])
        for entry in config["arena"]["camera_calibrations"]:
            entry["pixels_per_mm_projector"] = value
        del h5[CANVAS]
        h5.create_dataset(CANVAS, data=json.dumps(config), dtype=h5py.string_dtype())
    # 10 px/mm: radius 1 mm -> 10 px, arrow 3 mm -> 30 px.
    _edit_settings(
        path,
        lambda s: (
            s.update(pixels_per_mm_projector=value, radius_px=10.0),
            s["heading_arrow"].update(length_px=30.0),
        ),
    )


def _build(path: Path, commit: str, dirty: int = 0) -> None:
    with h5py.File(path, "r+") as h5:
        h5["/metadata/session"].attrs["git_commit_hash"] = commit
        h5["/metadata/session"].attrs["build_dirty"] = np.int32(dirty)


def test_canvas_file_scale_from_a_listed_build_is_read_and_recorded(tmp_path):
    from fisheye.shared.unified_h5.video_overlay import OVERLAY_SCALE_DEFECT

    path = _copy(tmp_path)
    _canvas_file_scale(path, 10.0)
    _build(path, DEFECT_BUILD)
    result = _validate(path)
    assert result["producer_defects"] == {
        OVERLAY_SCALE_DEFECT: {"citrus_build": DEFECT_BUILD, "scale_used": 10.0}
    }


@pytest.mark.parametrize(("commit", "dirty"), [("0" * 40, 0), (DEFECT_BUILD, 1)])
def test_canvas_file_scale_outside_the_listed_clean_builds_is_refused(tmp_path, commit, dirty):
    path = _copy(tmp_path)
    _canvas_file_scale(path, 10.0)
    _build(path, commit, dirty)
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_scale_authority_mismatch"):
        _validate(path)


def test_a_listed_build_still_needs_the_canvas_file_scale_exactly(tmp_path):
    path = _copy(tmp_path)
    _canvas_file_scale(path, 10.0)
    _edit_settings(path, lambda s: s.update(pixels_per_mm_projector=11.0, radius_px=11.0))
    _build(path, DEFECT_BUILD)
    with pytest.raises(UnifiedH5ContractError, match="video_overlay_scale_authority_mismatch"):
        _validate(path)


def test_a_listed_build_with_the_correct_scale_records_nothing(tmp_path):
    path = _copy(tmp_path)
    _build(path, DEFECT_BUILD)
    assert "producer_defects" not in _validate(path)
