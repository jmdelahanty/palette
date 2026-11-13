from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.chaser_behavior import (
    BEHAVIOR_CLASS_LABELS,
    ChaserBehaviorClass,
    behavior_class_label,
    canonical_behavior_label,
    resolve_configured_chaser_behaviors,
)
from fisheye.analysis.chaser_distance_runs import _refuse_programmed_chasers


def test_resolve_configured_chaser_behaviors_supports_variable_cardinality() -> None:
    payload = {
        "steps": [
            {
                "parameters": {
                    "chasers": [
                        {"enable_chase": True, "enable_random_movement": True},
                        {"enable_chase": False, "enable_random_movement": False},
                        {"enable_chase": False, "enable_random_movement": True},
                    ]
                }
            }
        ]
    }

    behaviors = resolve_configured_chaser_behaviors(payload)

    assert [behavior.chaser_index for behavior in behaviors] == [0, 1, 2]
    assert [behavior.behavior_class_id for behavior in behaviors] == [1, 3, 2]
    assert [behavior.behavior_class for behavior in behaviors] == [
        "aggressive",
        "inert",
        "random_non_chasing",
    ]


def test_canonical_behavior_label_maps_legacy_benign_to_inert() -> None:
    assert canonical_behavior_label("benign") == "inert"
    assert canonical_behavior_label("AGGRESSIVE") == "aggressive"


PROGRAM = {
    "schema_version": 2,
    "idle_motion": {"type": "wander", "speed_mm_per_sec": 30.0},
    "activation": {"type": "bernoulli_interval", "probability": 0.15, "interval_s": 1.0},
    "phases": [{"type": "chase", "speed_mm_per_sec": 60.0, "duration_s": 5.0}],
}


def _chasers(*chasers: dict) -> dict:
    return {"steps": [{"parameters": {"chasers": list(chasers)}}]}


def test_a_chaser_with_a_behavior_program_is_programmed_not_aggressive() -> None:
    # batman_composable shape: a program next to enable_chase true, which
    # Citrus ignores at runtime for a programmed chaser.
    behaviors = resolve_configured_chaser_behaviors(
        _chasers(
            {"behavior_mode": 0, "behavior_program": PROGRAM, "enable_chase": True},
            {"behavior_mode": 0, "behavior_program": PROGRAM, "enable_random_movement": True},
        )
    )
    assert [b.behavior_class_id for b in behaviors] == [4, 4]
    assert [b.behavior_class for b in behaviors] == ["programmed", "programmed"]


def test_a_passive_chaser_with_a_program_is_inert_as_citrus_resolves_it() -> None:
    (behavior,) = resolve_configured_chaser_behaviors(
        _chasers({"behavior_mode": 1, "behavior_program": PROGRAM, "enable_chase": True})
    )
    assert behavior.behavior_class == "inert"


def test_a_null_program_and_the_simple_forms_keep_their_classes() -> None:
    behaviors = resolve_configured_chaser_behaviors(
        _chasers(
            {"behavior_program": None, "enable_chase": True},
            {"enable_chase": True},
            {"enable_random_movement": True},
            {},
        )
    )
    assert [b.behavior_class for b in behaviors] == [
        "aggressive",
        "aggressive",
        "random_non_chasing",
        "inert",
    ]


def test_a_recorded_class_is_taken_as_recorded() -> None:
    (behavior,) = resolve_configured_chaser_behaviors(
        _chasers({"chaser_behavior_class_id": 4, "enable_chase": True})
    )
    assert behavior.behavior_class == "programmed"


@pytest.mark.parametrize("class_id", [5, -1, "x", None])
def test_an_unknown_class_id_is_refused(class_id) -> None:
    with pytest.raises(ValueError):
        behavior_class_label(class_id)
    with pytest.raises(ValueError):
        resolve_configured_chaser_behaviors(_chasers({"chaser_behavior_class_id": class_id}))


def test_the_vocabulary_is_zero_to_four() -> None:
    assert BEHAVIOR_CLASS_LABELS == {
        0: "unknown",
        1: "aggressive",
        2: "random_non_chasing",
        3: "inert",
        4: "programmed",
    }
    assert [behavior_class_label(i) for i in range(4)] == [
        "unknown",
        "aggressive",
        "random_non_chasing",
        "inert",
    ]


def test_chaser_distance_runs_refuse_programmed_chasers() -> None:
    _refuse_programmed_chasers(np.asarray([1, 3, 2, 0], dtype=np.int8))
    with pytest.raises(ValueError, match="programmed"):
        _refuse_programmed_chasers(
            np.asarray([1, int(ChaserBehaviorClass.PROGRAMMED)], dtype=np.int8)
        )
