"""Deterministic planning for declarative analysis DAGs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .contracts import AnalysisWorkflow, WorkflowNode


@dataclass(frozen=True)
class NodePlan:
    node_id: str
    kind: str
    stage_id: str | None
    action: str
    depends_on: tuple[str, ...]
    output_run_from: str | None
    artifact_path: str | None
    selected_run: str | None
    reason: str
    execution_policy: str | None
    temporal_policy: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "stage_id": self.stage_id,
            "action": self.action,
            "depends_on": list(self.depends_on),
            "output_run_from": self.output_run_from,
            "artifact_path": self.artifact_path,
            "selected_run": self.selected_run,
            "reason": self.reason,
            "execution_policy": self.execution_policy,
            "temporal_policy": dict(self.temporal_policy),
        }


@dataclass(frozen=True)
class WorkflowPlan:
    workflow_id: str
    targets: tuple[str, ...]
    ready: bool
    topological_order: tuple[str, ...]
    execution_order: tuple[str, ...]
    nodes: tuple[NodePlan, ...]
    temporal_policy: Mapping[str, object]

    @property
    def node_by_id(self) -> Mapping[str, NodePlan]:
        return MappingProxyType({node.node_id: node for node in self.nodes})

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_id": self.workflow_id,
            "targets": list(self.targets),
            "ready": self.ready,
            "topological_order": list(self.topological_order),
            "execution_order": list(self.execution_order),
            "temporal_policy": dict(self.temporal_policy),
            "nodes": [node.to_dict() for node in self.nodes],
        }


def topological_order(workflow: AnalysisWorkflow) -> tuple[str, ...]:
    """Return stable node order or raise on a dependency cycle."""

    nodes = workflow.node_by_id
    indegree = {node_id: 0 for node_id in nodes}
    downstream = {node_id: [] for node_id in nodes}
    for node in workflow.nodes:
        for dependency in node.depends_on:
            indegree[node.node_id] += 1
            downstream[dependency].append(node.node_id)
    insertion_order = {node.node_id: index for index, node in enumerate(workflow.nodes)}
    ready = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=insertion_order.__getitem__,
    )
    ordered: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_id)
        for child in sorted(downstream[node_id], key=insertion_order.__getitem__):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=insertion_order.__getitem__)
    if len(ordered) != len(nodes):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"analysis workflow contains a dependency cycle: {', '.join(cyclic)}")
    return tuple(ordered)


def _target_closure(
    workflow: AnalysisWorkflow,
    targets: Sequence[str],
) -> set[str]:
    nodes = workflow.node_by_id
    closure: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in closure:
            return
        if node_id not in nodes:
            raise ValueError(f"unknown workflow target: {node_id!r}")
        closure.add(node_id)
        for dependency in nodes[node_id].depends_on:
            visit(dependency)

    for target in targets:
        visit(str(target))
    return closure


def plan_analysis_workflow(
    workflow: AnalysisWorkflow,
    availability: Mapping[str, object],
    *,
    targets: Sequence[str] | None = None,
) -> WorkflowPlan:
    """Plan selected targets without executing or opening array payloads.

    ``availability`` is keyed by canonical stage id.  Values are expected to
    expose ``available``, ``artifact_path``, ``run_name``, and ``reason``
    attributes; keeping the planner structural makes it easy to supply registry
    or test-double availability later.
    """

    selected_targets = tuple(str(value) for value in (targets or workflow.targets))
    closure = _target_closure(workflow, selected_targets)
    ordered = tuple(node_id for node_id in topological_order(workflow) if node_id in closure)
    nodes = workflow.node_by_id
    planned: dict[str, NodePlan] = {}

    for node_id in ordered:
        node: WorkflowNode = nodes[node_id]
        stage_status = availability.get(node.stage_id) if node.stage_id is not None else None
        is_available = bool(getattr(stage_status, "available", False))
        blocked_dependencies = [
            dependency
            for dependency in node.depends_on
            if planned[dependency].action == "blocked"
        ]
        if is_available:
            action = "reuse"
            reason = str(getattr(stage_status, "reason", "persisted stage is available"))
        elif blocked_dependencies:
            action = "blocked"
            reason = "blocked by " + ", ".join(blocked_dependencies)
        elif not node.runnable:
            action = "blocked"
            stage_reason = str(
                getattr(
                    stage_status,
                    "reason",
                    "required persisted stage is unavailable",
                )
            )
            reason = stage_reason
        else:
            action = "run"
            if node.kind == "export":
                reason = "materialize configured export product"
            else:
                reason = str(getattr(stage_status, "reason", "persisted stage is unavailable"))

        selected_run = (
            str(getattr(stage_status, "run_name", "") or "") or None
            if stage_status is not None
            else None
        )
        if selected_run is None and node.stage_id is not None:
            selected_run = workflow.run_selection.get(node.stage_id)
        artifact_path = (
            str(getattr(stage_status, "artifact_path", "") or "") or None
            if stage_status is not None
            else None
        )
        planned[node_id] = NodePlan(
            node_id=node.node_id,
            kind=node.kind,
            stage_id=node.stage_id,
            action=action,
            depends_on=node.depends_on,
            output_run_from=node.output_run_from,
            artifact_path=artifact_path,
            selected_run=selected_run,
            reason=reason,
            execution_policy=node.execution_policy,
            temporal_policy=workflow.temporal_policy.product_policy(node.temporal_product),
        )

    target_ready = all(planned[target].action != "blocked" for target in selected_targets)
    execution_order = tuple(
        node_id for node_id in ordered if planned[node_id].action == "run"
    )
    return WorkflowPlan(
        workflow_id=workflow.workflow_id,
        targets=selected_targets,
        ready=target_ready,
        topological_order=ordered,
        execution_order=execution_order,
        nodes=tuple(planned[node_id] for node_id in ordered),
        temporal_policy=MappingProxyType(workflow.temporal_policy.to_dict()),
    )


__all__ = [
    "NodePlan",
    "WorkflowPlan",
    "plan_analysis_workflow",
    "topological_order",
]
