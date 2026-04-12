from __future__ import annotations

from fisheye.pose.schema import schema_from_package


def test_traditional_v1_schema_includes_heading_computation_metadata() -> None:
    schema = schema_from_package("traditional_v1")

    heading = schema.metadata["heading_computation"]
    assert heading["enabled"] is True
    assert heading["origin"] == {"op": "midpoint", "labels": ["eye_left", "eye_right"]}
    assert heading["direction_from"] == {"op": "keypoint", "label": "swim_bladder"}
    assert heading["direction_to"] == {"op": "midpoint", "labels": ["eye_left", "eye_right"]}
    assert heading["dependent_keypoints"] == ["swim_bladder", "eye_left", "eye_right"]


def test_traditional_v2_schema_limits_heading_dependencies_to_core_three_points() -> None:
    schema = schema_from_package("traditional_v2")

    heading = schema.metadata["heading_computation"]
    assert heading["enabled"] is True
    assert heading["dependent_keypoints"] == ["swim_bladder", "eye_left", "eye_right"]
    assert "snout_tip" not in heading["dependent_keypoints"]
    assert "tail_tip" not in heading["dependent_keypoints"]
