"""Shared pose schema utilities for defining keypoint topologies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


RUNTIME_COORD_DIMS = 2
_KEYPOINT_LABEL_ALIASES = {
    "bladder": "swim_bladder",
    "swim_bladder": "swim_bladder",
    "swim-bladder": "swim_bladder",
    "eye_left": "eye_left",
    "left_eye": "eye_left",
    "left-eye": "eye_left",
    "eye_right": "eye_right",
    "right_eye": "eye_right",
    "right-eye": "eye_right",
}
HEAD_TRIANGLE_KEYPOINT_LABELS = ("swim_bladder", "eye_left", "eye_right")


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


@dataclass(frozen=True)
class ResolvedSkeletonEdges:
    edge_pairs: tuple[tuple[int, int], ...]
    source: str


@dataclass(frozen=True)
class ResolvedHeadTriangleIndices:
    swim_bladder: int
    eye_left: int
    eye_right: int
    source: str

    @property
    def as_tuple(self) -> tuple[int, int, int]:
        return (self.swim_bladder, self.eye_left, self.eye_right)


def undirected_edge_topology(value: object) -> tuple[tuple[int, int], ...]:
    """Return an order- and direction-independent skeleton topology signature."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    normalized: list[tuple[int, int]] = []
    for edge in value:
        if (
            not isinstance(edge, Sequence)
            or isinstance(edge, (str, bytes, bytearray))
            or len(edge) != 2
        ):
            return ()
        try:
            source = int(edge[0])
            target = int(edge[1])
        except (TypeError, ValueError):
            return ()
        if source == target or source < 0 or target < 0:
            return ()
        normalized.append((min(source, target), max(source, target)))
    return tuple(sorted(normalized))


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


def canonicalize_keypoint_label(value: object) -> Optional[str]:
    text = _normalize_text(value)
    if text is None:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    return _KEYPOINT_LABEL_ALIASES.get(normalized, normalized)


def _normalize_label_sequence(values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    labels: list[str] = []
    for value in values:
        label = canonicalize_keypoint_label(value)
        if label is not None:
            labels.append(label)
    return tuple(labels)


def _extract_pose_schema_labels(pose_schema: object) -> tuple[str, ...]:
    if isinstance(pose_schema, PoseSchema):
        return tuple(
            label
            for label in (canonicalize_keypoint_label(node.name) for node in pose_schema.nodes)
            if label is not None
        )

    pose_schema_map = _coerce_mapping(pose_schema)
    if pose_schema_map is None:
        return ()

    labels = _normalize_label_sequence(pose_schema_map.get("keypoint_labels"))
    if labels:
        return labels

    nodes_meta = pose_schema_map.get("nodes")
    if isinstance(nodes_meta, Sequence) and not isinstance(nodes_meta, (str, bytes, bytearray)):
        labels_from_nodes: list[str] = []
        for node in nodes_meta:
            if isinstance(node, Mapping):
                label = canonicalize_keypoint_label(node.get("name"))
            else:
                label = canonicalize_keypoint_label(node)
            if label is not None:
                labels_from_nodes.append(label)
        if labels_from_nodes:
            return tuple(labels_from_nodes)

    node_names = _normalize_label_sequence(pose_schema_map.get("node_names"))
    if node_names:
        return node_names

    return ()


def resolve_keypoint_labels_from_attrs(
    attrs: Mapping[str, object],
    *,
    keypoint_count: Optional[int] = None,
) -> tuple[str, ...]:
    labels = _normalize_label_sequence(attrs.get("keypoint_labels"))
    if not labels:
        labels = _extract_pose_schema_labels(attrs.get("pose_schema"))
    if keypoint_count is not None and labels and len(labels) != int(keypoint_count):
        raise ValueError(
            f"Keypoint labels count {len(labels)} does not match runtime keypoint count {int(keypoint_count)}."
        )
    return labels


def resolve_required_keypoint_indices(
    labels: Sequence[object],
    required_labels: Sequence[object],
    *,
    keypoint_count: Optional[int] = None,
) -> dict[str, int]:
    normalized_labels = _normalize_label_sequence(labels)
    if keypoint_count is not None and len(normalized_labels) != int(keypoint_count):
        raise ValueError(
            f"Keypoint labels count {len(normalized_labels)} does not match runtime keypoint count {int(keypoint_count)}."
        )

    label_to_idx: dict[str, int] = {}
    for idx, label in enumerate(normalized_labels):
        label_to_idx.setdefault(label, idx)

    resolved: dict[str, int] = {}
    missing: list[str] = []
    for required in required_labels:
        canonical = canonicalize_keypoint_label(required)
        if canonical is None:
            continue
        if canonical not in label_to_idx:
            missing.append(canonical)
            continue
        resolved[canonical] = int(label_to_idx[canonical])

    if missing:
        raise ValueError(f"Required keypoint labels missing: {', '.join(missing)}")
    return resolved


def resolve_required_keypoint_indices_from_attrs(
    attrs: Mapping[str, object],
    required_labels: Sequence[object],
    *,
    keypoint_count: Optional[int] = None,
) -> dict[str, int]:
    labels = resolve_keypoint_labels_from_attrs(attrs, keypoint_count=keypoint_count)
    if not labels:
        raise ValueError("Keypoint labels are missing from run attrs and pose_schema metadata.")
    return resolve_required_keypoint_indices(
        labels,
        required_labels,
        keypoint_count=keypoint_count,
    )


def resolve_head_triangle_indices(
    labels: Sequence[object],
    *,
    keypoint_count: Optional[int] = None,
    allow_legacy_3point_fallback: bool = False,
) -> ResolvedHeadTriangleIndices:
    """Resolve the canonical swim-bladder/left-eye/right-eye triangle indices."""
    try:
        resolved = resolve_required_keypoint_indices(
            labels,
            HEAD_TRIANGLE_KEYPOINT_LABELS,
            keypoint_count=keypoint_count,
        )
    except ValueError:
        if allow_legacy_3point_fallback and keypoint_count == 3:
            return ResolvedHeadTriangleIndices(
                swim_bladder=0,
                eye_left=1,
                eye_right=2,
                source="legacy_3point_fallback",
            )
        raise
    return ResolvedHeadTriangleIndices(
        swim_bladder=int(resolved["swim_bladder"]),
        eye_left=int(resolved["eye_left"]),
        eye_right=int(resolved["eye_right"]),
        source="labels",
    )


def resolve_head_triangle_indices_from_attrs(
    attrs: Mapping[str, object],
    *,
    keypoint_count: Optional[int] = None,
    allow_legacy_3point_fallback: bool = False,
) -> ResolvedHeadTriangleIndices:
    """Resolve canonical head-triangle indices from run attrs or pose schema."""
    labels = resolve_keypoint_labels_from_attrs(attrs, keypoint_count=keypoint_count)
    try:
        return resolve_head_triangle_indices(
            labels,
            keypoint_count=keypoint_count,
            allow_legacy_3point_fallback=allow_legacy_3point_fallback,
        )
    except ValueError:
        if allow_legacy_3point_fallback and keypoint_count == 3 and not labels:
            return ResolvedHeadTriangleIndices(
                swim_bladder=0,
                eye_left=1,
                eye_right=2,
                source="legacy_3point_fallback",
            )
        raise


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


def _coerce_edge_pairs(raw_edges: object, *, n_keypoints: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes, bytearray)):
        return ()

    pairs: list[tuple[int, int]] = []
    for edge in raw_edges:
        a_raw: object | None = None
        b_raw: object | None = None
        if isinstance(edge, Mapping):
            a_raw = edge.get("from")
            if a_raw is None:
                a_raw = edge.get("source")
            if a_raw is None:
                a_raw = edge.get("a")
            b_raw = edge.get("to")
            if b_raw is None:
                b_raw = edge.get("target")
            if b_raw is None:
                b_raw = edge.get("b")
        elif isinstance(edge, Sequence) and not isinstance(edge, (str, bytes, bytearray)) and len(edge) >= 2:
            a_raw = edge[0]
            b_raw = edge[1]
        if a_raw is None or b_raw is None:
            continue
        try:
            a = int(a_raw)
            b = int(b_raw)
        except Exception:
            continue
        if 0 <= a < int(n_keypoints) and 0 <= b < int(n_keypoints) and a != b:
            left, right = (a, b) if a < b else (b, a)
            pair = (left, right)
            if pair not in pairs:
                pairs.append(pair)
    return tuple(pairs)


def normalize_ordered_skeleton_edges(
    raw_edges: object,
    *,
    n_keypoints: int,
    field: str = "skeleton edges",
) -> tuple[tuple[int, int], ...]:
    """Validate an exact directed, ordered skeleton edge list.

    Unlike :func:`undirected_edge_topology`, this representation is suitable
    for immutable training manifests: neither edge direction nor list order is
    normalized away. Reversed duplicates are rejected because they would
    describe the same undirected drawing edge twice while producing different
    manifest content.
    """
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an exact ordered edge list.")
    keypoint_count = int(n_keypoints)
    if keypoint_count <= 0:
        raise ValueError(f"{field} requires a positive keypoint count.")

    edges: list[tuple[int, int]] = []
    undirected_seen: set[tuple[int, int]] = set()
    for edge_index, edge in enumerate(raw_edges):
        if (
            not isinstance(edge, Sequence)
            or isinstance(edge, (str, bytes, bytearray))
            or len(edge) != 2
        ):
            raise ValueError(f"{field}[{edge_index}] must contain exactly two integer indices.")
        source_raw, target_raw = edge[0], edge[1]
        if (
            isinstance(source_raw, bool)
            or isinstance(target_raw, bool)
            or not isinstance(source_raw, int)
            or not isinstance(target_raw, int)
        ):
            raise ValueError(f"{field}[{edge_index}] must contain exactly two integer indices.")
        source = int(source_raw)
        target = int(target_raw)
        if source == target or not (0 <= source < keypoint_count and 0 <= target < keypoint_count):
            raise ValueError(
                f"{field}[{edge_index}]={list(edge)!r} is outside [0,{keypoint_count}) "
                "or is a self-edge."
            )
        undirected = (min(source, target), max(source, target))
        if undirected in undirected_seen:
            raise ValueError(f"{field}[{edge_index}] duplicates an existing undirected edge.")
        undirected_seen.add(undirected)
        edges.append((source, target))
    return tuple(edges)


def resolve_ordered_skeleton_edges_from_attrs(
    attrs: Mapping[str, object],
    *,
    n_keypoints: int,
    allow_package_schema: bool = True,
) -> ResolvedSkeletonEdges:
    """Resolve exact ordered edges without inventing a topology.

    Explicit ``pose_schema`` and ``keypoint_skeleton`` declarations must agree.
    A named packaged schema may supply omitted edges only when its cardinality
    and skeleton ID agree with the surrounding metadata. No legacy triangle is
    synthesized.
    """
    pose_schema = _coerce_mapping(attrs.get("pose_schema"))
    pose_edges_raw: object | None = None
    if pose_schema is not None:
        pose_edges_raw = pose_schema.get("edges")
        if pose_edges_raw is None:
            pose_edges_raw = pose_schema.get("skeleton")

    direct_edges_raw = attrs.get("keypoint_skeleton")
    pose_edges: tuple[tuple[int, int], ...] | None = None
    direct_edges: tuple[tuple[int, int], ...] | None = None
    if pose_edges_raw is not None:
        pose_edges = normalize_ordered_skeleton_edges(
            pose_edges_raw, n_keypoints=n_keypoints, field="pose_schema skeleton edges"
        )
    if direct_edges_raw is not None:
        direct_edges = normalize_ordered_skeleton_edges(
            direct_edges_raw, n_keypoints=n_keypoints, field="keypoint_skeleton"
        )
    if pose_edges is not None and direct_edges is not None and pose_edges != direct_edges:
        raise ValueError("pose_schema edges disagree with keypoint_skeleton exact ordering.")
    if pose_edges is not None:
        return ResolvedSkeletonEdges(edge_pairs=pose_edges, source="pose_schema")
    if direct_edges is not None:
        return ResolvedSkeletonEdges(edge_pairs=direct_edges, source="keypoint_skeleton")

    schema_name = _normalize_text(pose_schema.get("name")) if pose_schema is not None else None
    if schema_name is None:
        skeleton_id = _normalize_text(attrs.get("skeleton_id"))
        if skeleton_id is None and pose_schema is not None:
            skeleton_id = _normalize_text(pose_schema.get("skeleton_id"))
        if skeleton_id is not None and skeleton_id.startswith("pose_schema:"):
            schema_name = skeleton_id.split(":", 1)[1].strip() or None
    if not allow_package_schema or schema_name is None:
        return ResolvedSkeletonEdges(edge_pairs=(), source="none")

    try:
        package_schema = schema_from_package(schema_name)
    except FileNotFoundError:
        return ResolvedSkeletonEdges(edge_pairs=(), source="none")
    if package_schema.num_keypoints != int(n_keypoints):
        raise ValueError(
            f"Packaged pose schema {schema_name!r} has {package_schema.num_keypoints} keypoints, "
            f"not runtime count {int(n_keypoints)}."
        )
    package_skeleton_id = _normalize_text(package_schema.metadata.get("skeleton_id")) or (
        f"pose_schema:{package_schema.name}"
    )
    resolved_identity = resolve_skeleton_identity_from_attrs(attrs, keypoint_count=int(n_keypoints))
    if resolved_identity.skeleton_id is not None and resolved_identity.skeleton_id != package_skeleton_id:
        raise ValueError(
            f"Packaged pose schema {schema_name!r} skeleton_id {package_skeleton_id!r} "
            f"does not match declared {resolved_identity.skeleton_id!r}."
        )
    package_edges = normalize_ordered_skeleton_edges(
        package_schema.edges,
        n_keypoints=n_keypoints,
        field=f"packaged pose schema {schema_name!r} edges",
    )
    return ResolvedSkeletonEdges(
        edge_pairs=package_edges,
        source=f"pose_schema_package:{schema_name}",
    )


def resolve_skeleton_edges_from_attrs(
    attrs: Mapping[str, object],
    *,
    n_keypoints: int,
    allow_legacy_triangle: bool = True,
) -> ResolvedSkeletonEdges:
    """Resolve skeleton edge pairs from current metadata with a narrow legacy fallback."""

    pose_schema = _coerce_mapping(attrs.get("pose_schema"))
    raw_edges: object | None = None
    source = "none"

    if pose_schema is not None:
        raw_edges = pose_schema.get("edges")
        if raw_edges is None:
            raw_edges = pose_schema.get("skeleton")
        if raw_edges is not None:
            source = "pose_schema"

    if raw_edges is None:
        raw_edges = attrs.get("keypoint_skeleton")
        if raw_edges is not None:
            source = "keypoint_skeleton"

    if raw_edges is None and allow_legacy_triangle and int(n_keypoints) == 3:
        raw_edges = ((0, 1), (0, 2), (1, 2))
        source = "legacy_default_triangle"

    return ResolvedSkeletonEdges(
        edge_pairs=_coerce_edge_pairs(raw_edges, n_keypoints=int(n_keypoints)),
        source=source,
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
