from __future__ import annotations

from fisheye.analysis.chaser_behavior import (
    canonical_behavior_label,
    resolve_configured_chaser_behaviors,
)


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
