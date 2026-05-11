"""Inspect run-level lineage as a read-only DAG.

The canonical model is a pair of node and edge tables. Text, Mermaid, and DOT
renderings are projections of that same DAG, not independent sources of truth.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fisheye.shared.run_lineage_fingerprint import normalize_lineage_value
from fisheye.utils.audit_analysis_staleness import (
    RUN_PARENT_SPECS,
    SOURCE_RUN_ATTR_CANDIDATE_PARENTS,
    SourceAudit,
    _attrs_dict,
    _group_names,
    _latest_reference,
    _resolve_internal_path,
    audit_source_ref,
    discover_source_refs,
)
from fisheye.utils.zarr_io import open_zarr_root


LINEAGE_DAG_SCHEMA_ID = "palette.run_lineage_dag"
LINEAGE_DAG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LineageNode:
    node_id: str
    path: str
    family: str | None
    run_id: str | None
    exists: bool
    schema_id: str | None = None
    schema_version: int | None = None
    method: str | None = None
    method_version: str | None = None
    lineage_hash: str | None = None
    fingerprint_status: str | None = None
    latest_parent_path: str | None = None
    latest_run_id: str | None = None
    is_latest: bool | None = None


@dataclass(frozen=True)
class LineageEdge:
    source_node_id: str
    target_node_id: str
    edge_key: str
    source_path: str
    target_path: str
    status: str
    message: str
    expected_fingerprint: str | None = None
    actual_fingerprint: str | None = None
    actual_fingerprint_status: str | None = None


@dataclass(frozen=True)
class RunLineageGraph:
    schema_id: str
    schema_version: int
    zarr_path: str
    root_paths: list[str]
    nodes: list[LineageNode]
    edges: list[LineageEdge]


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_attr(attrs: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = attrs.get(name)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


def _parent_family_from_path(parent_path: tuple[str, ...]) -> str:
    name = parent_path[-1]
    if name.endswith("_runs"):
        return f"{name[:-5]}_run"
    if name.endswith("s"):
        return f"{name[:-1]}_run"
    return f"{name}_run"


def _known_run_parents() -> tuple[tuple[str, tuple[str, ...]], ...]:
    items: dict[tuple[str, ...], str] = {
        spec.parent_path: spec.family for spec in RUN_PARENT_SPECS
    }
    for parent_paths in SOURCE_RUN_ATTR_CANDIDATE_PARENTS.values():
        for parent_path in parent_paths:
            items.setdefault(parent_path, _parent_family_from_path(parent_path))
    return tuple(
        sorted(
            ((family, parent_path) for parent_path, family in items.items()),
            key=lambda item: (-len(item[1]), item[1]),
        )
    )


RUN_PARENTS = _known_run_parents()


def _normalize_path(path: str) -> str:
    return "/".join(part for part in str(path).strip().replace("\\", "/").split("/") if part)


def _owner_run(path: str) -> tuple[str | None, str, str | None]:
    parts = tuple(part for part in _normalize_path(path).split("/") if part)
    for family, parent_path in RUN_PARENTS:
        if len(parts) <= len(parent_path):
            continue
        if parts[: len(parent_path)] != parent_path:
            continue
        run_id = parts[len(parent_path)]
        run_path = "/".join((*parent_path, run_id))
        return family, run_path, run_id
    return None, "/".join(parts), parts[-1] if parts else None


def _node_for_path(root: Any, zarr_path: Path, path: str) -> LineageNode:
    family, run_path, run_id = _owner_run(path)
    group = _resolve_internal_path(root, run_path, archive_path=zarr_path)
    attrs = _attrs_dict(group) if group is not None else {}
    latest = _latest_reference(root, run_path)
    latest_parent_path = latest[0] if latest else None
    latest_run_id = latest[1] if latest else None
    referenced_run_id = latest[2] if latest else run_id
    is_latest = None
    if latest_run_id is not None and referenced_run_id is not None:
        is_latest = latest_run_id == referenced_run_id
    return LineageNode(
        node_id=run_path,
        path=run_path,
        family=family,
        run_id=run_id,
        exists=group is not None,
        schema_id=_string_attr(attrs, "schema_id"),
        schema_version=_coerce_int(attrs.get("schema_version") or attrs.get("output_schema_version")),
        method=_string_attr(attrs, "method", "detection_method"),
        method_version=_string_attr(attrs, "method_version"),
        lineage_hash=_string_attr(attrs, "lineage_hash", "source_lineage_hash", "source_fingerprint"),
        fingerprint_status=_string_attr(attrs, "fingerprint_status", "lineage_fingerprint_status"),
        latest_parent_path=latest_parent_path,
        latest_run_id=latest_run_id,
        is_latest=is_latest,
    )


def _all_analysis_run_paths(root: Any, run_families: set[str] | None) -> list[str]:
    run_paths: list[str] = []
    for spec in RUN_PARENT_SPECS:
        if run_families is not None and spec.family not in run_families:
            continue
        parent = root
        try:
            for part in spec.parent_path:
                if part not in parent:
                    parent = None
                    break
                parent = parent[part]
        except Exception:
            parent = None
        if parent is None:
            continue
        for run_id in _group_names(parent):
            run_paths.append("/".join((*spec.parent_path, run_id)))
    return sorted(run_paths)


def build_run_lineage_graph(
    zarr_path: Path,
    *,
    root_paths: Sequence[str] | None = None,
    run_families: set[str] | None = None,
    require_latest_sources: bool = False,
) -> RunLineageGraph:
    """Build a run-lineage DAG from one Palette Zarr archive.

    Edges point from upstream source runs to downstream derived runs.
    """

    zarr_path = Path(zarr_path)
    root = open_zarr_root(zarr_path, mode="r")
    if root_paths:
        roots = [_owner_run(_normalize_path(path))[1] for path in root_paths]
    else:
        roots = _all_analysis_run_paths(root, run_families)

    nodes: dict[str, LineageNode] = {}
    edges: dict[tuple[str, str, str, str], LineageEdge] = {}
    visited: set[str] = set()
    active: set[str] = set()

    def add_node(path: str) -> LineageNode:
        _, run_path, _ = _owner_run(path)
        if run_path not in nodes:
            nodes[run_path] = _node_for_path(root, zarr_path, run_path)
        return nodes[run_path]

    def add_edge(target_path: str, source_audit: SourceAudit) -> None:
        _, source_run_path, _ = _owner_run(source_audit.path)
        add_node(source_run_path)
        key = (
            source_run_path,
            target_path,
            source_audit.key,
            source_audit.path,
        )
        edges[key] = LineageEdge(
            source_node_id=source_run_path,
            target_node_id=target_path,
            edge_key=source_audit.key,
            source_path=source_audit.path,
            target_path=target_path,
            status=source_audit.status,
            message=source_audit.message,
            expected_fingerprint=source_audit.expected_fingerprint,
            actual_fingerprint=source_audit.actual_fingerprint,
            actual_fingerprint_status=source_audit.actual_fingerprint_status,
        )

    def visit(run_path: str) -> None:
        add_node(run_path)
        if run_path in visited or run_path in active:
            return
        active.add(run_path)
        group = _resolve_internal_path(root, run_path, archive_path=zarr_path)
        if group is not None:
            for ref in discover_source_refs(root, group):
                source_audit = audit_source_ref(
                    root,
                    ref,
                    zarr_path=zarr_path,
                    require_latest_sources=require_latest_sources,
                )
                add_edge(run_path, source_audit)
                _, source_run_path, _ = _owner_run(source_audit.path)
                if source_run_path not in active:
                    visit(source_run_path)
        active.remove(run_path)
        visited.add(run_path)

    for root_path in roots:
        visit(root_path)

    return RunLineageGraph(
        schema_id=LINEAGE_DAG_SCHEMA_ID,
        schema_version=LINEAGE_DAG_SCHEMA_VERSION,
        zarr_path=str(zarr_path),
        root_paths=list(dict.fromkeys(roots)),
        nodes=[nodes[key] for key in sorted(nodes)],
        edges=[
            edges[key]
            for key in sorted(
                edges,
                key=lambda item: (item[1], item[0], item[2], item[3]),
            )
        ],
    )


def graph_to_json_dict(graph: RunLineageGraph) -> dict[str, Any]:
    return normalize_lineage_value(asdict(graph))


def render_json(graph: RunLineageGraph) -> str:
    return json.dumps(graph_to_json_dict(graph), indent=2, allow_nan=False, sort_keys=True)


def _node_label(node: LineageNode) -> str:
    suffixes: list[str] = []
    if node.family:
        suffixes.append(node.family)
    if node.is_latest is False:
        suffixes.append("not_latest")
    if not node.exists:
        suffixes.append("missing")
    suffix = f" [{', '.join(suffixes)}]" if suffixes else ""
    return f"{node.path}{suffix}"


def render_tree(graph: RunLineageGraph) -> str:
    nodes = {node.node_id: node for node in graph.nodes}
    incoming: dict[str, list[LineageEdge]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target_node_id, []).append(edge)
    for target_edges in incoming.values():
        target_edges.sort(key=lambda edge: (edge.edge_key, edge.source_node_id, edge.source_path))

    lines: list[str] = [
        f"{graph.zarr_path}: run lineage DAG",
        "edge direction: source -> target",
    ]
    shown: set[str] = set()

    def walk(node_id: str, prefix: str, active: set[str]) -> None:
        if node_id in active:
            lines.append(f"{prefix}(cycle) {node_id}")
            return
        if node_id in shown:
            lines.append(f"{prefix}(already shown) {node_id}")
            return
        shown.add(node_id)
        active = set(active)
        active.add(node_id)
        source_edges = incoming.get(node_id, [])
        for index, edge in enumerate(source_edges):
            branch = "`- " if index == len(source_edges) - 1 else "+- "
            child_prefix = "   " if index == len(source_edges) - 1 else "|  "
            source_node = nodes.get(edge.source_node_id)
            label = _node_label(source_node) if source_node is not None else edge.source_node_id
            lines.append(
                f"{prefix}{branch}{edge.edge_key}: {label} "
                f"[{edge.status}]"
            )
            if edge.source_path != edge.source_node_id:
                lines.append(f"{prefix}{child_prefix}source_path: {edge.source_path}")
            if edge.status != "fresh":
                lines.append(f"{prefix}{child_prefix}message: {edge.message}")
            walk(edge.source_node_id, prefix + child_prefix, active)

    for index, root_path in enumerate(graph.root_paths):
        if index:
            lines.append("")
        root_node = nodes.get(root_path)
        lines.append(_node_label(root_node) if root_node is not None else root_path)
        walk(root_path, "", set())
    return "\n".join(lines)


def _graph_node_name(index: int) -> str:
    return f"n{index}"


def _escape_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def render_mermaid(graph: RunLineageGraph) -> str:
    node_names = {node.node_id: _graph_node_name(index) for index, node in enumerate(graph.nodes)}
    lines = ["flowchart TD"]
    for node in graph.nodes:
        label_parts = [node.path]
        if node.family:
            label_parts.append(node.family)
        if not node.exists:
            label_parts.append("missing")
        elif node.is_latest is False:
            label_parts.append("not latest")
        label = "<br/>".join(_escape_label(part) for part in label_parts)
        lines.append(f'  {node_names[node.node_id]}["{label}"]')
    for edge in graph.edges:
        label = _escape_label(f"{edge.edge_key}: {edge.status}")
        source = node_names.get(edge.source_node_id, edge.source_node_id)
        target = node_names.get(edge.target_node_id, edge.target_node_id)
        lines.append(f"  {source} -->|{label}| {target}")
    return "\n".join(lines)


def render_dot(graph: RunLineageGraph) -> str:
    node_names = {node.node_id: _graph_node_name(index) for index, node in enumerate(graph.nodes)}
    lines = ["digraph palette_run_lineage {", "  rankdir=LR;"]
    for node in graph.nodes:
        label_parts = [node.path]
        if node.family:
            label_parts.append(node.family)
        if not node.exists:
            label_parts.append("missing")
        elif node.is_latest is False:
            label_parts.append("not latest")
        label = "\\n".join(_escape_label(part) for part in label_parts)
        lines.append(f'  {node_names[node.node_id]} [label="{label}"];')
    for edge in graph.edges:
        label = _escape_label(f"{edge.edge_key}: {edge.status}")
        source = node_names.get(edge.source_node_id, edge.source_node_id)
        target = node_names.get(edge.target_node_id, edge.target_node_id)
        lines.append(f'  {source} -> {target} [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def render_graph(graph: RunLineageGraph, *, output_format: str) -> str:
    if output_format == "json":
        return render_json(graph)
    if output_format == "tree":
        return render_tree(graph)
    if output_format == "mermaid":
        return render_mermaid(graph)
    if output_format == "dot":
        return render_dot(graph)
    raise ValueError(f"unknown lineage graph format: {output_format}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a read-only Palette run-lineage DAG for one analysis Zarr.",
    )
    parser.add_argument("zarr_path", type=Path, help="Palette analysis Zarr archive")
    parser.add_argument(
        "--root",
        action="append",
        dest="root_paths",
        help="root run path to render upstream from; may be repeated. Defaults to all known analysis runs.",
    )
    parser.add_argument(
        "--run-family",
        action="append",
        choices=sorted({spec.family for spec in RUN_PARENT_SPECS}),
        help="when --root is omitted, limit discovered roots to one run family; may be repeated",
    )
    parser.add_argument(
        "--format",
        choices=("tree", "json", "mermaid", "dot"),
        default="tree",
        help="output format",
    )
    parser.add_argument(
        "--require-latest-sources",
        action="store_true",
        help="treat source refs that do not point at parent latest as stale",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    graph = build_run_lineage_graph(
        args.zarr_path,
        root_paths=args.root_paths,
        run_families=set(args.run_family) if args.run_family else None,
        require_latest_sources=bool(args.require_latest_sources),
    )
    print(render_graph(graph, output_format=str(args.format)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
