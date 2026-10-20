"""Reactive phototaxis states invariants (admission revision 6, phototaxis_v1).

Each case writes a small, internally consistent file from the four rows of
Citrus's phototaxis_v1 witness (one pose update, then three lost-pose holds),
then breaks one invariant.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.phototaxis import STATES, validate_phototaxis_states

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "citrus_unified_h5_phototaxis_v1"
    / "phototaxis_v1_complete.h5"
)
CAMERA = "CAM-42"
STEP = 0
REACTIVE = {
    "camera_view_assumed": "dorsal_unmirrored",
    "fill_style": "two_tone",
    "heading_definition_ref": "agent-contracts:citrus-palette/heading/heading_convention_v1.json",
    "hysteresis_frame": "camera",
    "left_side": "side_a",
    "maximum_held_heading_updates": 60,
    "minimum_heading_displacement_degrees": 5.0,
    "minimum_keypoint_confidence": 0.5,
    "module": "PhototaxisHeadingSplit",
    "side_rule_id": "projected_left_axis_v1",
    "unknown_heading_fallback": "uniform_side_b",
}


def _witness():
    with h5py.File(FIXTURE, "r") as h5:
        return (
            h5[STATES][()],
            h5["/observations/pose/updates"][()],
            h5["/observations/pose/objects"][()],
            json.loads(h5["/observations/pose/contract_json"][()]),
        )


WITNESS_ROWS, WITNESS_UPDATES, WITNESS_OBJECTS, WITNESS_CONTRACT = _witness()


def _snapshot(reactive=REACTIVE, step=STEP):
    steps = [{"step_index": index, "features": {}} for index in range(3)]
    steps[step]["features"] = {  # trial-index step 0 in the witness
        "fill": {"type": "heading_split", "style": "two_tone"},
        "reactive": reactive,
    }
    return SimpleNamespace(trial_index_payload={"steps": steps})


def _execution(step=STEP, start=0, end=10):
    interval = SimpleNamespace(
        start_stimulus_frame_inclusive=start, end_stimulus_frame_exclusive=end
    )
    return SimpleNamespace(steps=(SimpleNamespace(step_index=step, interval=interval),))


def _consistent_rows() -> np.ndarray:
    rows = WITNESS_ROWS.copy()
    rows["protocol_step_index"] = STEP
    return rows


def _write(
    path: Path,
    rows: np.ndarray,
    *,
    pose_version: int = 2,
    pose_heading=None,
    states=True,
    pose_gate=0.5,
    shaman_camera=None,
    pose_camera=None,
) -> Path:
    contract = json.loads(json.dumps(WITNESS_CONTRACT))
    contract["schema_version"] = pose_version
    contract["minimum_keypoint_confidence"] = pose_gate
    first_pose = int(np.flatnonzero(np.append(rows["input_used"] == 1, True))[0]) % len(rows)
    camera = int(rows["source_camera_id"][first_pose])
    updates = WITNESS_UPDATES.copy()
    updates["sequence_id"] = rows["pose_sequence_id"][first_pose]
    updates["camera_id"] = camera if pose_camera is None else pose_camera
    objects = WITNESS_OBJECTS.copy()
    angle, x, y = pose_heading or (
        rows["heading_camera_angle_degrees"][first_pose],
        *rows["heading_origin_camera_px"][first_pose],
    )
    objects["heading_status"] = 7
    objects["heading_angle_degrees"] = angle
    objects["heading_origin_x_px"] = x
    objects["heading_origin_y_px"] = y
    with h5py.File(path, "w") as h5:
        if states:
            h5.create_dataset(STATES, data=rows)
        pose = h5.create_group("/observations/pose")
        pose.create_dataset("contract_json", data=json.dumps(contract), dtype=h5py.string_dtype())
        pose.create_dataset("updates", data=updates)
        pose.create_dataset("objects", data=objects)
        binding = {"shaman_numeric_camera_id": camera if shaman_camera is None else shaman_camera}
        h5.create_dataset(
            "/correspondence/acquisition/binding_json", data=json.dumps(binding), dtype=h5py.string_dtype()
        )
    return path


def _validate(path: Path, *, snapshot=None, execution=None) -> int:
    with h5py.File(path, "r") as h5:
        return validate_phototaxis_states(
            h5,
            snapshot or _snapshot(),
            execution or _execution(),
            camera_serial=CAMERA,
        )


def test_consistent_witness_rows_are_admitted(tmp_path):
    rows = _consistent_rows()
    assert rows["input_used"].tolist() == [1, 2, 2, 2]
    assert _validate(_write(tmp_path / "ok.h5", rows)) == 4


def _mutate(**fields):
    def apply(rows):
        for name, value in fields.items():
            index, field = name.split("__")
            rows[field][int(index)] = value
        return rows

    return apply


def _fallback_row(rows):
    rows = rows[1:2].copy()
    for name in ("held_camera_angle_degrees", "held_origin_canvas_px", "held_direction_canvas",
                 "held_left_canvas", "held_angle_degrees_canvas"):
        rows[name] = 0
    rows["held_valid"] = 0
    rows["input_used"] = 3
    rows["fallback_id"] = 3
    # A step's first update starts with no held heading: age 1, as Citrus
    # writes it (every step-first fallback row in the 10-17/18 recordings).
    rows["held_age_updates"] = 1
    return rows


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_mutate(**{"0__heading_valid": 2}), "phototaxis_validity_not_boolean:heading_valid"),
        (_mutate(**{"1__heading_camera_angle_degrees": 1.0}), "phototaxis_invalid_payload_nonzero:heading_camera_angle_degrees"),
        (_mutate(**{"0__heading_status": 10}), "phototaxis_heading_status_unknown"),
        (_mutate(**{"0__heading_status": 5}), "phototaxis_heading_valid_status"),
        (_mutate(**{"0__input_used": 0}), "phototaxis_input_used_unknown"),
        (_mutate(**{"0__fallback_id": 4}), "phototaxis_fallback_unknown"),
        (_mutate(**{"1__fallback_id": 3}), "phototaxis_fallback_input"),
        # Without overlay states a step's first row shows freshness only by its
        # age, so a pose row aged 1 there reads as a stale re-read (and the
        # next row's count no longer follows).
        (_mutate(**{"0__held_age_updates": 1}), "phototaxis_age_not_counted|phototaxis_pose_input_without_fresh_heading"),
        (_mutate(**{"1__held_age_updates": 0}), "phototaxis_age_not_counted"),
        (_mutate(**{"1__held_age_updates": 61}), "phototaxis_age_not_counted"),
        (_mutate(**{"3__held_age_updates": 5}), "phototaxis_age_not_counted"),
        (_mutate(**{"0__held_angle_degrees_canvas": 37.0}), "phototaxis_pose_held_differs:held_angle_degrees_canvas"),
        (_mutate(**{"0__protocol_step_index": 1}), "phototaxis_step_not_active"),
        (_mutate(**{"0__pose_model_hash64": 1}), "phototaxis_pose_identity_mismatch"),
        (_mutate(**{"1__source_camera_id": 8, "1__source_camera_id_valid": 1}), "phototaxis_source_camera_not_unique"),
        (_mutate(**{"1__stimulus_frame_num": 2}), "phototaxis_states_frame_order"),
    ],
)
def test_invariant_violations_are_refused(tmp_path, mutate, reason):
    path = _write(tmp_path / "bad.h5", mutate(_consistent_rows()))
    with pytest.raises(UnifiedH5ContractError, match=reason):
        _validate(path)


def test_hysteresis_hold_must_stay_under_the_threshold(tmp_path):
    rows = _consistent_rows()
    # Row 1 becomes a hysteresis hold: a usable raw heading 6 degrees away.
    for raw, held in (("midline_origin_canvas_px", "held_origin_canvas_px"),
                      ("midline_direction_canvas", "held_direction_canvas"),
                      ("midline_left_canvas", "held_left_canvas"),
                      ("midline_angle_degrees_canvas", "held_angle_degrees_canvas")):
        rows[raw][1] = rows[held][1]
    rows["heading_origin_camera_px"][1] = rows["heading_origin_camera_px"][0]
    rows["heading_status"][1] = 7
    rows["heading_valid"][1] = 1
    rows["midline_valid"][1] = 1
    # A hysteresis hold reads a new slot (fresh), so its age is 0 and the
    # count restarts after it.
    rows["pose_sequence_id"][1] = rows["pose_sequence_id"][0] + 1
    rows["held_age_updates"][1:] = (0, 1, 2)
    rows["heading_camera_angle_degrees"][1] = rows["held_camera_angle_degrees"][1] + 4.0
    assert _validate(_write(tmp_path / "hold.h5", rows)) == 4
    rows["heading_camera_angle_degrees"][1] = rows["held_camera_angle_degrees"][1] + 6.0
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_hysteresis_threshold"):
        _validate(_write(tmp_path / "jump.h5", rows))


def test_fallback_must_be_the_step_fallback(tmp_path):
    rows = _fallback_row(_consistent_rows())
    assert _validate(_write(tmp_path / "fallback.h5", rows)) == 1
    rows["fallback_id"] = 1
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_fallback_differs_from_step"):
        _validate(_write(tmp_path / "other.h5", rows))


def test_hold_beyond_the_step_limit_is_refused(tmp_path):
    path = _write(tmp_path / "limit.h5", _consistent_rows())  # held ages 1, 2, 3
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_hold_beyond_limit"):
        _validate(path, snapshot=_snapshot({**REACTIVE, "maximum_held_heading_updates": 2}))


def test_hold_never_has_no_age_limit(tmp_path):
    path = _write(tmp_path / "never.h5", _consistent_rows())
    _validate(path, snapshot=_snapshot({**REACTIVE, "maximum_held_heading_updates": "never"}))
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_hold_beyond_limit"):
        _validate(path, snapshot=_snapshot({**REACTIVE, "maximum_held_heading_updates": 0}))


def test_pose_input_needs_its_pose_row(tmp_path):
    path = _write(tmp_path / "pose.h5", _consistent_rows(), pose_heading=(1.0, 20.0, 10.0))
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_pose_row_missing"):
        _validate(path)


def test_reactive_files_need_the_table_and_pose_v2(tmp_path):
    with pytest.raises(UnifiedH5ContractError, match="reactive_phototaxis_states_missing"):
        _validate(_write(tmp_path / "none.h5", _consistent_rows(), states=False))
    with pytest.raises(UnifiedH5ContractError, match="reactive_phototaxis_requires_pose_v2"):
        _validate(_write(tmp_path / "v1.h5", _consistent_rows(), pose_version=1))


def test_states_without_a_reactive_step_are_refused(tmp_path):
    path = _write(tmp_path / "static.h5", _consistent_rows())
    static = SimpleNamespace(trial_index_payload={"steps": [{"step_index": 0, "features": {}}]})
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_states_without_reactive_step"):
        _validate(path, snapshot=static)
    assert _validate(_write(tmp_path / "empty.h5", _consistent_rows(), states=False), snapshot=static) == 0


def test_rows_in_the_overlap_frame_follow_their_named_step(tmp_path):
    # Citrus: intervals overlap by one frame (old end - 1 == new start); a step's
    # first row is at start + 1. protocol_step_index decides, not uniqueness.
    rows = _consistent_rows()
    def interval(start, end):
        return SimpleNamespace(start_stimulus_frame_inclusive=start, end_stimulus_frame_exclusive=end)

    execution = SimpleNamespace(
        steps=(
            SimpleNamespace(step_index=1, interval=interval(0, 1)),
            SimpleNamespace(step_index=STEP, interval=interval(0, 10)),
        )
    )
    assert _validate(_write(tmp_path / "overlap.h5", rows), execution=execution) == 4
    late = SimpleNamespace(steps=(SimpleNamespace(step_index=STEP, interval=interval(1, 10)),))
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_step_not_active"):
        _validate(_write(tmp_path / "late.h5", rows), execution=late)


def test_a_fallback_first_row_is_valid(tmp_path):
    rows = _consistent_rows()
    first = _fallback_row(rows)
    # The pose update after it read a new slot.
    first["pose_sequence_id"] = rows["pose_sequence_id"][0] - 1
    later = rows.copy()
    later["stimulus_frame_num"] += 1
    combined = np.concatenate([first, later])
    combined["stimulus_frame_num"][0] = 0
    assert _validate(_write(tmp_path / "first.h5", combined)) == 5


def test_a_step_gate_below_the_pose_gate_is_refused(tmp_path):
    path = _write(tmp_path / "gate.h5", _consistent_rows(), pose_gate=0.6)
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_step_gate_below_pose_gate"):
        _validate(path)


def test_source_camera_must_be_the_binding_and_pose_slot_camera(tmp_path):
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_source_camera_not_binding"):
        _validate(_write(tmp_path / "binding.h5", _consistent_rows(), shaman_camera=3))
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_source_camera_not_pose_slot"):
        _validate(_write(tmp_path / "slot.h5", _consistent_rows(), pose_camera=3))


def test_a_fresh_usable_heading_restarts_the_age(tmp_path):
    rows = _consistent_rows()
    # Row 1 reads a new slot with a usable heading 4 degrees from the held one:
    # a hysteresis hold, so its age must be 0.
    for raw, held in (("midline_origin_canvas_px", "held_origin_canvas_px"),
                      ("midline_direction_canvas", "held_direction_canvas"),
                      ("midline_left_canvas", "held_left_canvas"),
                      ("midline_angle_degrees_canvas", "held_angle_degrees_canvas")):
        rows[raw][1] = rows[held][1]
    rows["heading_origin_camera_px"][1] = rows["heading_origin_camera_px"][0]
    rows["heading_status"][1] = 7
    rows["heading_valid"][1] = 1
    rows["midline_valid"][1] = 1
    rows["heading_camera_angle_degrees"][1] = rows["held_camera_angle_degrees"][1] + 4.0
    rows["pose_sequence_id"][1] = rows["pose_sequence_id"][0] + 1
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_age_on_fresh_heading"):
        _validate(_write(tmp_path / "aged.h5", rows))  # ages still 1, 2, 3
    # The same usable heading re-read from the same slot is stale: age grows.
    rows["pose_sequence_id"][1] = rows["pose_sequence_id"][0]
    assert _validate(_write(tmp_path / "stale.h5", rows)) == 4


def test_a_fallback_never_coincides_with_a_fresh_usable_heading(tmp_path):
    rows = _consistent_rows()
    rows["input_used"][0] = 3
    rows["fallback_id"][0] = 3
    for name in ("held_camera_angle_degrees", "held_origin_canvas_px", "held_direction_canvas",
                 "held_left_canvas", "held_angle_degrees_canvas"):
        rows[name][0] = 0
    rows["held_valid"][0] = 0
    with pytest.raises(UnifiedH5ContractError, match="phototaxis_fallback_with_fresh_heading"):
        _validate(_write(tmp_path / "fallback.h5", rows))
