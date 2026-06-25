from __future__ import annotations

import numpy as np
import zarr
from zarr.storage import MemoryStore

from fisheye.pose.metric_schema import (
    compute_derived_metric_results,
    metric_schema_from_package,
    resolve_metric_schema_for_group,
)
from fisheye.pose.schema import schema_from_package


def test_traditional_v3_schema_loads_with_expected_nodes_and_edges() -> None:
    schema = schema_from_package("traditional_v3")

    assert schema.name == "traditional_v3"
    assert schema.node_names == [
        "swim_bladder",
        "eye_left",
        "eye_right",
        "snout_tip",
        "tail_tip",
        "mid_tail",
        "right_pectoral_fin_insertion",
        "right_pectoral_fin_tip",
        "left_pectoral_fin_insertion",
        "left_pectoral_fin_tip",
    ]
    assert schema.edges == [
        [0, 1],
        [0, 2],
        [1, 2],
        [3, 1],
        [3, 2],
        [0, 5],
        [5, 4],
        [2, 6],
        [0, 6],
        [6, 7],
        [1, 8],
        [0, 8],
        [8, 9],
    ]
    assert schema.metadata["version"] == 3
    assert schema.metadata["skeleton_id"] == "pose_skel_traditional_v3"
    assert schema.metadata["migration_from"] == "traditional_v2"


def test_traditional_v3_heading_stays_on_core_head_triangle() -> None:
    schema = schema_from_package("traditional_v3")

    heading = schema.metadata["heading_computation"]
    assert heading["enabled"] is True
    assert heading["dependent_keypoints"] == ["swim_bladder", "eye_left", "eye_right"]
    assert "snout_tip" not in heading["dependent_keypoints"]
    assert "tail_tip" not in heading["dependent_keypoints"]
    assert "mid_tail" not in heading["dependent_keypoints"]


def test_traditional_v3_metric_schema_resolves_and_computes_distances() -> None:
    schema = metric_schema_from_package("traditional_v3")
    labels = [
        "swim_bladder",
        "eye_left",
        "eye_right",
        "snout_tip",
        "tail_tip",
        "mid_tail",
        "right_pectoral_fin_insertion",
        "right_pectoral_fin_tip",
        "left_pectoral_fin_insertion",
        "left_pectoral_fin_tip",
    ]
    points = np.array(
        [
            [5.0, 5.0],  # swim_bladder
            [8.0, 4.0],  # eye_left
            [8.0, 6.0],  # eye_right
            [10.0, 5.0],  # snout_tip
            [0.0, 5.0],  # tail_tip
            [2.5, 5.0],  # mid_tail
            [6.0, 7.0],  # right_pectoral_fin_insertion
            [5.0, 9.0],  # right_pectoral_fin_tip
            [6.0, 3.0],  # left_pectoral_fin_insertion
            [5.0, 1.0],  # left_pectoral_fin_tip
        ],
        dtype=np.float64,
    )

    result = compute_derived_metric_results(
        points,
        keypoint_labels=labels,
        schema=schema,
        roi_diagonal=10.0,
    )

    assert schema.schema_name == "traditional_v3_derived_metrics"
    assert schema.skeleton_id == "pose_skel_traditional_v3"
    assert schema.metric_labels == [
        "total_length",
        "tail_length",
        "head_length",
        "eye_span",
        "anterior_tail_segment",
        "posterior_tail_segment",
        "right_pectoral_fin_length",
        "left_pectoral_fin_length",
        "right_pectoral_insertion_to_eye",
        "left_pectoral_insertion_to_eye",
    ]
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                10.0,
                5.0,
                5.0,
                2.0,
                2.5,
                2.5,
                np.sqrt(5.0),
                np.sqrt(5.0),
                np.sqrt(5.0),
                np.sqrt(5.0),
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(result.values_norm, result.values / 10.0)
    np.testing.assert_array_equal(result.valid, np.ones((10,), dtype=bool))


def test_resolve_metric_schema_for_group_uses_traditional_v3_pose_schema() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = root.create_group("refined_keypoints_runs").create_group("refined_v3")
    run.attrs["pose_schema"] = {
        "name": "traditional_v3",
        "skeleton_id": "pose_skel_traditional_v3",
    }

    schema = resolve_metric_schema_for_group(run, required=True)

    assert schema is not None
    assert schema.schema_name == "traditional_v3_derived_metrics"
    assert schema.metric_labels[-4:] == [
        "right_pectoral_fin_length",
        "left_pectoral_fin_length",
        "right_pectoral_insertion_to_eye",
        "left_pectoral_insertion_to_eye",
    ]
