"""Pose schema utilities for defining keypoint topologies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Node:
    id: int
    name: str


@dataclass(frozen=True)
class PoseSchema:
    name: str
    nodes: List[Node]
    edges: List[List[int]]
    metadata: Dict[str, object]

    @property
    def num_keypoints(self) -> int:
        return len(self.nodes)

    @property
    def node_names(self) -> List[str]:
        return [n.name for n in self.nodes]

    def index(self, node_name: str) -> int:
        for node in self.nodes:
            if node.name == node_name:
                return node.id
        raise ValueError(f"Node '{node_name}' not found in schema '{self.name}'")


def load_schema(schema_path: Path) -> PoseSchema:
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    nodes = [Node(id=node["id"], name=node["name"]) for node in raw["nodes"]]
    edges = raw.get("edges", [])
    metadata = raw.get("metadata", {})
    return PoseSchema(name=raw["name"], nodes=nodes, edges=edges, metadata=metadata)


def schema_from_package(name: str, base_dir: Optional[Path] = None) -> PoseSchema:
    module_path = Path(__file__).resolve()
    if len(module_path.parents) < 4:
        raise RuntimeError("Unexpected package layout; unable to resolve 'configs/fisheye/pose_schemas'.")
    default_dir = module_path.parents[3] / "configs" / "fisheye" / "pose_schemas"

    search_dirs = []
    if base_dir is not None:
        search_dirs.append(Path(base_dir))
    search_dirs.append(default_dir)

    tried_paths = []
    for directory in search_dirs:
        schema_path = directory / f"{name}.json"
        tried_paths.append(schema_path)
        if schema_path.exists():
            return load_schema(schema_path)

    tried_str = ", ".join(str(path) for path in tried_paths)
    raise FileNotFoundError(f"Pose schema '{name}' not found. Tried: {tried_str}")


def schema_from_metadata(metadata: Dict[str, object]) -> PoseSchema:
    name = metadata.get("name", "unknown_schema")
    nodes_meta = metadata.get("nodes") or metadata.get("node_names")
    if nodes_meta is None:
        raise ValueError("pose_schema metadata missing 'nodes' or 'node_names'")
    nodes: List[Node]
    if isinstance(nodes_meta, list) and nodes_meta and isinstance(nodes_meta[0], dict):

        nodes = [Node(id=node["id"], name=node["name"]) for node in nodes_meta]
    else:

        nodes = [Node(id=i, name=str(name)) for i, name in enumerate(nodes_meta)]
    edges = metadata.get("edges", [])
    meta = metadata.get("metadata", {})
    return PoseSchema(name=name, nodes=nodes, edges=edges, metadata=meta)
