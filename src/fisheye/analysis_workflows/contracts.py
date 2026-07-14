"""Contracts for configurable analysis-workflow DAGs.

The workflow contract is intentionally declarative.  It describes dependency
ordering and temporal output policies without granting permission to execute a
stage or mutate a recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from fisheye.registry.stage_catalog import canonical_stage_id, get_stage_spec


ANALYSIS_WORKFLOW_SCHEMA_ID = "palette.analysis_workflow"
ANALYSIS_WORKFLOW_SCHEMA_VERSION = 1
FRAMEWISE_RESOLUTION = "framewise"
NODE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
NODE_KINDS = frozenset({"prerequisite", "analysis", "export"})
TEMPORAL_PRODUCTS = frozenset(
    {"kinematics", "activity_spatial", "eye_traces", "tail_traces"}
)


def _positive_finite(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return parsed


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of strings")
    out = tuple(str(item).strip() for item in value)
    if any(not item for item in out):
        raise ValueError(f"{label} cannot contain empty values")
    return out


@dataclass(frozen=True)
class TemporalPolicy:
    """Portable temporal-resolution policy for analysis exports.

    Kinematics and summary bin sizes are configurable.  Eye and tail traces
    are fixed to framewise resolution because downsampling those authorities
    would discard the temporal signal needed by downstream analyses.
    """

    kinematics_sample_rate_hz: float = 10.0
    activity_spatial_bin_size_s: float = 5.0
    eye_trace_resolution: str = FRAMEWISE_RESOLUTION
    tail_trace_resolution: str = FRAMEWISE_RESOLUTION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kinematics_sample_rate_hz",
            _positive_finite(
                self.kinematics_sample_rate_hz,
                label="kinematics.sample_rate_hz",
            ),
        )
        object.__setattr__(
            self,
            "activity_spatial_bin_size_s",
            _positive_finite(
                self.activity_spatial_bin_size_s,
                label="activity_spatial.bin_size_s",
            ),
        )
        for label, value in (
            ("eye_traces.resolution", self.eye_trace_resolution),
            ("tail_traces.resolution", self.tail_trace_resolution),
        ):
            if str(value).strip().lower() != FRAMEWISE_RESOLUTION:
                raise ValueError(f"{label} must remain {FRAMEWISE_RESOLUTION!r}")
        object.__setattr__(self, "eye_trace_resolution", FRAMEWISE_RESOLUTION)
        object.__setattr__(self, "tail_trace_resolution", FRAMEWISE_RESOLUTION)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "TemporalPolicy":
        if raw is not None and not isinstance(raw, Mapping):
            raise ValueError("temporal_policy must be a mapping")
        data = dict(raw or {})
        unknown_sections = sorted(
            set(data) - {"kinematics", "activity_spatial", "eye_traces", "tail_traces"}
        )
        if unknown_sections:
            raise ValueError(
                "unknown temporal_policy section(s): " + ", ".join(unknown_sections)
            )

        sections: dict[str, Mapping[str, Any]] = {}
        allowed_fields = {
            "kinematics": {"sample_rate_hz"},
            "activity_spatial": {"bin_size_s"},
            "eye_traces": {"resolution"},
            "tail_traces": {"resolution"},
        }
        for section_name, allowed in allowed_fields.items():
            section = data.get(section_name, {})
            if section is None:
                section = {}
            if not isinstance(section, Mapping):
                raise ValueError(f"temporal_policy.{section_name} must be a mapping")
            unknown_fields = sorted(set(section) - allowed)
            if unknown_fields:
                raise ValueError(
                    f"unknown temporal_policy.{section_name} field(s): "
                    + ", ".join(unknown_fields)
                )
            sections[section_name] = section

        kinematics = sections["kinematics"]
        summaries = sections["activity_spatial"]
        eyes = sections["eye_traces"]
        tail = sections["tail_traces"]
        return cls(
            kinematics_sample_rate_hz=kinematics.get("sample_rate_hz", 10.0),
            activity_spatial_bin_size_s=summaries.get("bin_size_s", 5.0),
            eye_trace_resolution=eyes.get("resolution", FRAMEWISE_RESOLUTION),
            tail_trace_resolution=tail.get("resolution", FRAMEWISE_RESOLUTION),
        )

    def with_overrides(
        self,
        *,
        kinematics_sample_rate_hz: float | None = None,
        activity_spatial_bin_size_s: float | None = None,
    ) -> "TemporalPolicy":
        return replace(
            self,
            kinematics_sample_rate_hz=(
                self.kinematics_sample_rate_hz
                if kinematics_sample_rate_hz is None
                else kinematics_sample_rate_hz
            ),
            activity_spatial_bin_size_s=(
                self.activity_spatial_bin_size_s
                if activity_spatial_bin_size_s is None
                else activity_spatial_bin_size_s
            ),
        )

    def product_policy(self, product: str | None) -> Mapping[str, object]:
        if product is None:
            return MappingProxyType({})
        if product == "kinematics":
            return MappingProxyType(
                {
                    "resolution": "sampled",
                    "sample_rate_hz": self.kinematics_sample_rate_hz,
                    "source_authority": "framewise_zarr",
                }
            )
        if product == "activity_spatial":
            return MappingProxyType(
                {
                    "resolution": "fixed_time_bins",
                    "bin_size_s": self.activity_spatial_bin_size_s,
                    "source_authority": "framewise_zarr",
                }
            )
        if product == "eye_traces":
            return MappingProxyType(
                {
                    "resolution": self.eye_trace_resolution,
                    "source_authority": "framewise_zarr",
                }
            )
        if product == "tail_traces":
            return MappingProxyType(
                {
                    "resolution": self.tail_trace_resolution,
                    "source_authority": "framewise_zarr",
                }
            )
        raise KeyError(f"unknown temporal product: {product!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "kinematics": {
                "resolution": "sampled",
                "sample_rate_hz": self.kinematics_sample_rate_hz,
                "source_authority": "framewise_zarr",
            },
            "activity_spatial": {
                "resolution": "fixed_time_bins",
                "bin_size_s": self.activity_spatial_bin_size_s,
                "source_authority": "framewise_zarr",
            },
            "eye_traces": {
                "resolution": self.eye_trace_resolution,
                "source_authority": "framewise_zarr",
            },
            "tail_traces": {
                "resolution": self.tail_trace_resolution,
                "source_authority": "framewise_zarr",
            },
        }


@dataclass(frozen=True)
class WorkflowNode:
    """One prerequisite, analysis, or export-product node in a workflow."""

    node_id: str
    kind: str
    depends_on: tuple[str, ...] = ()
    stage_id: str | None = None
    runnable: bool = True
    temporal_product: str | None = None
    execution_policy: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        if not NODE_ID_PATTERN.fullmatch(node_id):
            raise ValueError(f"invalid workflow node id: {self.node_id!r}")
        object.__setattr__(self, "node_id", node_id)
        kind = str(self.kind).strip().lower()
        if kind not in NODE_KINDS:
            raise ValueError(f"node {node_id!r} has unsupported kind {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        dependencies = tuple(str(value).strip() for value in self.depends_on)
        if any(not NODE_ID_PATTERN.fullmatch(value) for value in dependencies):
            raise ValueError(f"node {node_id!r} has invalid dependency node id")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"node {node_id!r} repeats a dependency")
        object.__setattr__(self, "depends_on", dependencies)
        if self.stage_id is not None:
            object.__setattr__(self, "stage_id", canonical_stage_id(str(self.stage_id)))
        if kind in {"prerequisite", "analysis"} and self.stage_id is None:
            raise ValueError(f"node {node_id!r} requires a canonical stage_id")
        if not isinstance(self.runnable, bool):
            raise ValueError(f"node {node_id!r} runnable must be a boolean")
        if self.temporal_product is not None:
            object.__setattr__(self, "temporal_product", str(self.temporal_product).strip())
        if self.temporal_product is not None and self.temporal_product not in TEMPORAL_PRODUCTS:
            raise ValueError(
                f"node {node_id!r} has unknown temporal_product {self.temporal_product!r}"
            )
        if self.temporal_product is not None and kind != "export":
            raise ValueError(f"only export nodes may declare temporal_product ({node_id!r})")
        object.__setattr__(
            self,
            "execution_policy",
            str(self.execution_policy).strip() if self.execution_policy is not None else None,
        )
        object.__setattr__(self, "description", str(self.description).strip())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WorkflowNode":
        return cls(
            node_id=str(raw.get("id") or ""),
            kind=str(raw.get("kind") or "analysis"),
            depends_on=_string_tuple(raw.get("depends_on"), label="node.depends_on"),
            stage_id=(str(raw["stage_id"]) if raw.get("stage_id") is not None else None),
            runnable=raw.get("runnable", True),
            temporal_product=(
                str(raw["temporal_product"])
                if raw.get("temporal_product") is not None
                else None
            ),
            execution_policy=(
                str(raw["execution_policy"])
                if raw.get("execution_policy") is not None
                else None
            ),
            description=str(raw.get("description") or ""),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.node_id,
            "kind": self.kind,
            "stage_id": self.stage_id,
            "depends_on": list(self.depends_on),
            "runnable": self.runnable,
            "temporal_product": self.temporal_product,
            "execution_policy": self.execution_policy,
            "description": self.description,
        }


@dataclass(frozen=True)
class AnalysisWorkflow:
    """Validated analysis DAG plus its temporal and artifact-selection policy."""

    workflow_id: str
    description: str
    nodes: tuple[WorkflowNode, ...]
    targets: tuple[str, ...]
    temporal_policy: TemporalPolicy = field(default_factory=TemporalPolicy)
    run_selection: Mapping[str, str] = field(default_factory=dict)
    schema_id: str = ANALYSIS_WORKFLOW_SCHEMA_ID
    schema_version: int = ANALYSIS_WORKFLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_id != ANALYSIS_WORKFLOW_SCHEMA_ID:
            raise ValueError(
                f"unsupported analysis workflow schema_id {self.schema_id!r}; "
                f"expected {ANALYSIS_WORKFLOW_SCHEMA_ID!r}"
            )
        schema_version = int(self.schema_version)
        if schema_version != ANALYSIS_WORKFLOW_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported analysis workflow schema_version {self.schema_version!r}"
            )
        object.__setattr__(self, "schema_version", schema_version)
        workflow_id = str(self.workflow_id).strip()
        if not NODE_ID_PATTERN.fullmatch(workflow_id):
            raise ValueError(f"invalid workflow_id: {self.workflow_id!r}")
        object.__setattr__(self, "workflow_id", workflow_id)
        object.__setattr__(self, "description", str(self.description).strip())
        nodes = tuple(self.nodes)
        targets = tuple(str(value).strip() for value in self.targets)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "targets", targets)
        node_by_id = {node.node_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ValueError("workflow node ids must be unique")
        for node in self.nodes:
            unknown = sorted(set(node.depends_on) - set(node_by_id))
            if unknown:
                raise ValueError(
                    f"node {node.node_id!r} depends on unknown node(s): {', '.join(unknown)}"
                )
            if node.node_id in node.depends_on:
                raise ValueError(f"node {node.node_id!r} cannot depend on itself")
        unknown_targets = sorted(set(self.targets) - set(node_by_id))
        if unknown_targets:
            raise ValueError(f"unknown workflow target(s): {', '.join(unknown_targets)}")
        if not self.targets:
            raise ValueError("analysis workflow must declare at least one target")

        stage_node_by_id = {
            node.stage_id: node for node in self.nodes if node.stage_id is not None
        }
        stage_node_count = sum(node.stage_id is not None for node in self.nodes)
        if len(stage_node_by_id) != stage_node_count:
            raise ValueError("canonical stage ids may appear only once in a workflow")
        for node in self.nodes:
            if node.stage_id is None:
                continue
            declared_stage_dependencies = {
                node_by_id[dependency].stage_id
                for dependency in node.depends_on
                if node_by_id[dependency].stage_id is not None
            }
            required_in_profile = {
                dependency
                for dependency in get_stage_spec(node.stage_id).depends_on
                if dependency in stage_node_by_id
            }
            missing = sorted(required_in_profile - declared_stage_dependencies)
            if missing:
                raise ValueError(
                    f"node {node.node_id!r} omits catalog dependency node(s): "
                    f"{', '.join(missing)}"
                )

        normalized_selection: dict[str, str] = {}
        known_stages = {node.stage_id for node in self.nodes if node.stage_id is not None}
        for raw_stage, raw_run in dict(self.run_selection).items():
            stage_id = canonical_stage_id(str(raw_stage))
            if stage_id not in known_stages:
                raise ValueError(
                    f"run_selection references stage {stage_id!r} outside the workflow"
                )
            run_name = str(raw_run).strip()
            if not run_name:
                raise ValueError(f"run_selection for {stage_id!r} cannot be empty")
            normalized_selection[stage_id] = run_name
        object.__setattr__(self, "run_selection", MappingProxyType(normalized_selection))

        # Import lazily to keep the contracts module independent of planner types.
        from .dag import topological_order

        topological_order(self)

    @property
    def node_by_id(self) -> Mapping[str, WorkflowNode]:
        return MappingProxyType({node.node_id: node for node in self.nodes})

    def with_temporal_overrides(
        self,
        *,
        kinematics_sample_rate_hz: float | None = None,
        activity_spatial_bin_size_s: float | None = None,
    ) -> "AnalysisWorkflow":
        return replace(
            self,
            temporal_policy=self.temporal_policy.with_overrides(
                kinematics_sample_rate_hz=kinematics_sample_rate_hz,
                activity_spatial_bin_size_s=activity_spatial_bin_size_s,
            ),
        )

    def with_run_selection(self, updates: Mapping[str, str]) -> "AnalysisWorkflow":
        merged = dict(self.run_selection)
        merged.update(updates)
        return replace(self, run_selection=merged)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AnalysisWorkflow":
        node_rows = raw.get("nodes")
        if isinstance(node_rows, str) or not isinstance(node_rows, Sequence):
            raise ValueError("analysis workflow nodes must be a sequence")
        nodes = tuple(
            WorkflowNode.from_mapping(row)
            for row in node_rows
            if isinstance(row, Mapping)
        )
        if len(nodes) != len(node_rows):
            raise ValueError("every analysis workflow node must be a mapping")
        run_selection = raw.get("run_selection") or {}
        if not isinstance(run_selection, Mapping):
            raise ValueError("run_selection must be a mapping")
        return cls(
            schema_id=str(raw.get("schema_id") or ""),
            schema_version=int(raw.get("schema_version") or 0),
            workflow_id=str(raw.get("workflow_id") or ""),
            description=str(raw.get("description") or ""),
            nodes=nodes,
            targets=_string_tuple(raw.get("targets"), label="targets"),
            temporal_policy=TemporalPolicy.from_mapping(raw.get("temporal_policy")),
            run_selection={str(key): str(value) for key, value in run_selection.items()},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "description": self.description,
            "targets": list(self.targets),
            "temporal_policy": self.temporal_policy.to_dict(),
            "run_selection": dict(self.run_selection),
            "nodes": [node.to_dict() for node in self.nodes],
        }


def load_analysis_workflow(path: str | Path) -> AnalysisWorkflow:
    profile_path = Path(path).expanduser()
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"analysis workflow profile must contain a mapping: {profile_path}")
    return AnalysisWorkflow.from_mapping(payload)


def default_core_behavior_profile_path() -> Path:
    return Path(__file__).resolve().parent / "profiles" / "core_behavior_v1.yaml"


__all__ = [
    "ANALYSIS_WORKFLOW_SCHEMA_ID",
    "ANALYSIS_WORKFLOW_SCHEMA_VERSION",
    "AnalysisWorkflow",
    "FRAMEWISE_RESOLUTION",
    "TemporalPolicy",
    "WorkflowNode",
    "default_core_behavior_profile_path",
    "load_analysis_workflow",
]
