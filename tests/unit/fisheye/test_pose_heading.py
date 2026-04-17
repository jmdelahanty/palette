from __future__ import annotations

import math

import numpy as np

from fisheye.pose.heading import (
    compute_heading_from_attrs,
    compute_heading_from_spec,
    compute_heading_origin_from_spec,
    dependent_keypoints,
    evaluate_point_expression,
    resolve_heading_computation_from_attrs,
)
from fisheye.pose.schema import schema_from_package


def test_resolve_heading_computation_prefers_override() -> None:
    attrs = {
        "pose_schema": {
            "name": "custom",
            "metadata": {
                "heading_computation": {"version": 1, "enabled": True},
            },
        },
        "heading_computation_override": {"version": 1, "enabled": False},
        "heading_computation": {"version": 1, "enabled": True},
    }

    resolved = resolve_heading_computation_from_attrs(attrs)

    assert resolved.source == "heading_computation_override"
    assert resolved.spec == {"version": 1, "enabled": False}


def test_compute_heading_from_spec_uses_pose_metadata_and_ignores_non_heading_points() -> None:
    schema = schema_from_package("traditional_v2")
    points = np.array(
        [
            [10.0, 10.0],
            [20.0, 12.0],
            [20.0, 8.0],
            [999.0, 999.0],
            [-999.0, -999.0],
        ],
        dtype=np.float64,
    )

    heading = compute_heading_from_spec(
        schema.metadata["heading_computation"],
        labels=schema.node_names,
        points=points,
        strict=True,
    )

    assert math.isclose(heading, 0.0, abs_tol=1e-6)


def test_compute_heading_from_attrs_returns_nan_when_override_disables_heading() -> None:
    schema = schema_from_package("traditional_v1")
    attrs = {
        "pose_schema": {
            "name": schema.name,
            "nodes": [{"id": node.id, "name": node.name} for node in schema.nodes],
            "edges": schema.edges,
            "metadata": schema.metadata,
        },
        "heading_computation_override": {"version": 1, "enabled": False},
    }
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.float64,
    )

    heading = compute_heading_from_attrs(
        attrs,
        labels=["swim_bladder", "eye_left", "eye_right"],
        points=points,
    )

    assert math.isnan(heading)


def test_evaluate_point_expression_midpoint_returns_expected_xy() -> None:
    point = evaluate_point_expression(
        {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
        labels=["swim_bladder", "eye_left", "eye_right"],
        points=np.array(
            [
                [0.0, 0.0],
                [4.0, 2.0],
                [6.0, 6.0],
            ],
            dtype=np.float64,
        ),
        strict=True,
    )

    np.testing.assert_allclose(point, np.array([5.0, 4.0], dtype=np.float64))


def test_compute_heading_origin_and_dependencies_follow_explicit_spec() -> None:
    schema = schema_from_package("traditional_v1")
    spec = schema.metadata["heading_computation"]
    points = np.array(
        [
            [0.0, 0.0],
            [4.0, 2.0],
            [6.0, 6.0],
        ],
        dtype=np.float64,
    )

    origin = compute_heading_origin_from_spec(
        spec,
        labels=schema.node_names,
        points=points,
        strict=True,
    )

    np.testing.assert_allclose(origin, np.array([5.0, 4.0], dtype=np.float64))
    assert dependent_keypoints(spec) == ("swim_bladder", "eye_left", "eye_right")


def test_compute_heading_from_spec_returns_nan_for_missing_required_label() -> None:
    spec = {
        "version": 1,
        "enabled": True,
        "direction_from": {"op": "keypoint", "label": "swim_bladder"},
        "direction_to": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
        "origin": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
        "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"],
    }
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    heading = compute_heading_from_spec(
        spec,
        labels=["swim_bladder", "eye_left"],
        points=points,
    )

    assert math.isnan(heading)
