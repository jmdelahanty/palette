from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fisheye.utils import inspect_run_lineage_graph as lineage


class FakeGroup:
    def __init__(
        self,
        attrs: dict[str, Any] | None = None,
        children: dict[str, "FakeGroup"] | None = None,
    ) -> None:
        self.attrs = attrs or {}
        self.children = children or {}

    def __contains__(self, key: str) -> bool:
        return key in self.children

    def __getitem__(self, key: str) -> "FakeGroup":
        return self.children[key]

    def group_keys(self) -> list[str]:
        return list(self.children)


def _build_graph(root: FakeGroup, monkeypatch, tmp_path: Path, **kwargs: Any) -> lineage.RunLineageGraph:
    monkeypatch.setattr(lineage, "open_zarr_root", lambda path, mode="r": root)
    return lineage.build_run_lineage_graph(tmp_path / "archive.zarr", **kwargs)


def test_graph_builds_node_edge_tables_from_source_refs(monkeypatch, tmp_path: Path) -> None:
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "subject_shape_runs": FakeGroup(
                        attrs={"latest": "shape_1"},
                        children={
                            "shape_1": FakeGroup(
                                {
                                    "schema_id": "analysis.subject_shape_runs",
                                    "lineage_hash": "shapehash",
                                    "fingerprint_status": "complete",
                                }
                            )
                        },
                    ),
                    "eye_angle_runs": FakeGroup(
                        children={
                            "eye_1": FakeGroup(
                                {
                                    "schema_id": "analysis.eye_angle_runs",
                                    "source_refs": {
                                        "subject_shape": {
                                            "path": "analysis/subject_shape_runs/shape_1",
                                            "fingerprint": "shapehash",
                                        }
                                    },
                                }
                            )
                        }
                    ),
                }
            )
        }
    )

    graph = _build_graph(
        root,
        monkeypatch,
        tmp_path,
        root_paths=["analysis/eye_angle_runs/eye_1"],
    )

    assert [node.node_id for node in graph.nodes] == [
        "analysis/eye_angle_runs/eye_1",
        "analysis/subject_shape_runs/shape_1",
    ]
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.source_node_id == "analysis/subject_shape_runs/shape_1"
    assert edge.target_node_id == "analysis/eye_angle_runs/eye_1"
    assert edge.edge_key == "subject_shape"
    assert edge.status == "fresh"

    payload = json.loads(lineage.render_json(graph))
    assert payload["schema_id"] == lineage.LINEAGE_DAG_SCHEMA_ID
    assert payload["edges"][0]["actual_fingerprint"] == "shapehash"


def test_graph_maps_compact_table_source_to_owner_run(monkeypatch, tmp_path: Path) -> None:
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "swim_bout_runs": FakeGroup(
                        children={
                            "bouts_1": FakeGroup(
                                children={
                                    "tables": FakeGroup(
                                        children={
                                            "bouts": FakeGroup(
                                                {
                                                    "lineage_hash": "bouttable",
                                                    "fingerprint_status": "complete",
                                                }
                                            )
                                        }
                                    )
                                }
                            )
                        }
                    ),
                    "bout_kinematics_runs": FakeGroup(
                        children={
                            "bk_1": FakeGroup(
                                {
                                    "source_refs": {
                                        "source_swim_bout_path": (
                                            "analysis/swim_bout_runs/bouts_1/"
                                            "tables/bouts?candidate_id=0&signal_id=4"
                                        )
                                    },
                                    "source_fingerprints": {
                                        "source_swim_bout_path": "bouttable",
                                    },
                                }
                            )
                        }
                    ),
                }
            )
        }
    )

    graph = _build_graph(
        root,
        monkeypatch,
        tmp_path,
        root_paths=["analysis/bout_kinematics_runs/bk_1"],
    )

    assert {node.node_id for node in graph.nodes} == {
        "analysis/bout_kinematics_runs/bk_1",
        "analysis/swim_bout_runs/bouts_1",
    }
    assert graph.edges[0].source_node_id == "analysis/swim_bout_runs/bouts_1"
    assert graph.edges[0].source_path == "analysis/swim_bout_runs/bouts_1/tables/bouts"
    assert graph.edges[0].status == "fresh"


def test_graph_preserves_missing_source_edges(monkeypatch, tmp_path: Path) -> None:
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "subject_shape_runs": FakeGroup(children={}),
                    "eye_angle_runs": FakeGroup(
                        children={
                            "eye_1": FakeGroup(
                                {
                                    "source_refs": {
                                        "subject_shape": "analysis/subject_shape_runs/missing_shape"
                                    }
                                }
                            )
                        }
                    ),
                }
            )
        }
    )

    graph = _build_graph(
        root,
        monkeypatch,
        tmp_path,
        root_paths=["analysis/eye_angle_runs/eye_1"],
    )

    source = next(node for node in graph.nodes if node.node_id == "analysis/subject_shape_runs/missing_shape")
    assert source.exists is False
    assert graph.edges[0].status == "missing_source"


def test_tree_projection_marks_shared_nodes(monkeypatch, tmp_path: Path) -> None:
    shape_ref = {
        "path": "analysis/subject_shape_runs/shape_1",
        "fingerprint": "shapehash",
    }
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "subject_shape_runs": FakeGroup(
                        children={
                            "shape_1": FakeGroup(
                                {
                                    "lineage_hash": "shapehash",
                                    "fingerprint_status": "complete",
                                }
                            )
                        }
                    ),
                    "eye_angle_runs": FakeGroup(
                        children={"eye_1": FakeGroup({"source_refs": {"shape": shape_ref}})}
                    ),
                    "tail_kinematics_runs": FakeGroup(
                        children={"tail_1": FakeGroup({"source_refs": {"shape": shape_ref}})}
                    ),
                }
            )
        }
    )

    graph = _build_graph(
        root,
        monkeypatch,
        tmp_path,
        root_paths=[
            "analysis/eye_angle_runs/eye_1",
            "analysis/tail_kinematics_runs/tail_1",
        ],
    )

    text = lineage.render_tree(graph)
    assert "edge direction: source -> target" in text
    assert "(already shown) analysis/subject_shape_runs/shape_1" in text


def test_mermaid_and_dot_are_rendered_from_same_graph(monkeypatch, tmp_path: Path) -> None:
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "track_kinematics_runs": FakeGroup(
                        children={
                            "offline": FakeGroup(
                                children={"tk_1": FakeGroup({"lineage_hash": "tkhash"})}
                            )
                        }
                    ),
                    "swim_bout_runs": FakeGroup(
                        children={
                            "bouts_1": FakeGroup(
                                {
                                    "source_track_kinematics_run": "tk_1",
                                    "source_fingerprints": {
                                        "source_track_kinematics_run": "tkhash",
                                    },
                                }
                            )
                        }
                    ),
                }
            )
        }
    )

    graph = _build_graph(
        root,
        monkeypatch,
        tmp_path,
        root_paths=["analysis/swim_bout_runs/bouts_1"],
    )

    mermaid = lineage.render_mermaid(graph)
    dot = lineage.render_dot(graph)
    assert mermaid.startswith("flowchart TD")
    assert "source_track_kinematics_run: fresh" in mermaid
    assert dot.startswith("digraph palette_run_lineage")
    assert "source_track_kinematics_run: fresh" in dot
