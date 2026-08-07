from __future__ import annotations

from fisheye.pose.schema import schema_from_package, undirected_edge_topology


def test_traditional_v2_schema_loads_with_expected_nodes_and_edges() -> None:
    schema = schema_from_package("traditional_v2")

    assert schema.name == "traditional_v2"
    assert schema.node_names == [
        "swim_bladder",
        "eye_left",
        "eye_right",
        "snout_tip",
        "tail_tip",
    ]
    assert schema.edges == [
        [0, 1],
        [0, 2],
        [1, 2],
        [3, 1],
        [3, 2],
        [0, 4],
    ]
    assert schema.metadata["version"] == 2
    assert schema.metadata["skeleton_id"] == "pose_skel_traditional_v2"


def test_skeleton_topology_comparison_ignores_edge_direction_and_order() -> None:
    package_edges = [[0, 1], [0, 2], [1, 2], [3, 1], [3, 2], [0, 4]]
    manifest_edges = [[0, 4], [2, 3], [1, 3], [2, 1], [2, 0], [1, 0]]

    assert undirected_edge_topology(package_edges) == undirected_edge_topology(
        manifest_edges
    )
