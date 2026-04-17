"""Pose schema utilities for defining keypoint topologies."""

from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


RUNTIME_COORD_DIMS = 2


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


@dataclass(frozen=True)
class ResolvedSkeletonIdentity:
    pose_schema_name: Optional[str]
    skeleton_id: Optional[str]
    kpt_shape: Optional[tuple[int, int]]


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _coerce_mapping(value: object) -> Optional[dict[str, object]]:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def normalize_kpt_shape(value: object) -> Optional[tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            first = int(value[0])
            second = int(value[1])
        except (TypeError, ValueError):
            return None
        if first > 0 and second > 0:
            return (first, second)
    return None


def schema_to_attr_payload(
    schema: PoseSchema,
    *,
    coord_dims: int = RUNTIME_COORD_DIMS,
) -> dict[str, object]:
    metadata = dict(schema.metadata)
    skeleton_id = _normalize_text(metadata.get("skeleton_id")) or f"pose_schema:{schema.name}"
    labels = list(schema.node_names)
    return {
        "name": schema.name,
        "skeleton_id": skeleton_id,
        "kpt_shape": [int(schema.num_keypoints), int(coord_dims)],
        "keypoint_labels": labels,
        "nodes": [{"id": int(node.id), "name": str(node.name)} for node in schema.nodes],
        "edges": [[int(src), int(dst)] for src, dst in schema.edges],
        "metadata": metadata,
        "source": f"configs/fisheye/pose_schemas/{schema.name}.json",
    }


def schema_payload_from_package(
    name: str,
    *,
    base_dir: Optional[Path] = None,
    coord_dims: int = RUNTIME_COORD_DIMS,
) -> tuple[PoseSchema, dict[str, object]]:
    schema = schema_from_package(name, base_dir=base_dir)
    return schema, schema_to_attr_payload(schema, coord_dims=coord_dims)


def resolve_skeleton_identity_from_attrs(
    attrs: Mapping[str, object],
    *,
    keypoint_count: Optional[int] = None,
    coord_dims: Optional[int] = None,
) -> ResolvedSkeletonIdentity:
    pose_schema = _coerce_mapping(attrs.get("pose_schema"))
    pose_schema_name = _normalize_text(pose_schema.get("name")) if pose_schema is not None else None

    skeleton_id = _normalize_text(attrs.get("skeleton_id"))
    if skeleton_id is None and pose_schema is not None:
        skeleton_id = _normalize_text(pose_schema.get("skeleton_id"))
    if skeleton_id is None and pose_schema_name is not None:
        skeleton_id = f"pose_schema:{pose_schema_name}"

    kpt_shape = normalize_kpt_shape(attrs.get("kpt_shape"))
    if kpt_shape is None and pose_schema is not None:
        kpt_shape = normalize_kpt_shape(pose_schema.get("kpt_shape"))
    if kpt_shape is None and keypoint_count is not None:
        resolved_dims = coord_dims
        if resolved_dims is None and pose_schema is not None:
            pose_shape = normalize_kpt_shape(pose_schema.get("kpt_shape"))
            if pose_shape is not None:
                resolved_dims = int(pose_shape[1])
        if resolved_dims is None or int(resolved_dims) <= 0:
            resolved_dims = RUNTIME_COORD_DIMS
        kpt_shape = (int(keypoint_count), int(resolved_dims))

    return ResolvedSkeletonIdentity(
        pose_schema_name=pose_schema_name,
        skeleton_id=skeleton_id,
        kpt_shape=kpt_shape,
    )


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
