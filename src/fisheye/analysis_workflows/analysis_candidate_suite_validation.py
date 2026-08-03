"""Live family binding for derived-analysis execution benchmark suites.

The benchmark-suite envelope proves that a storage plan is internally
consistent.  It does not, by itself, prove that a caller used the maintained
logical schema for the named family.  These validators reconstruct the exact
family declarations from the suite's observed facts and compare them to the
persisted plan declarations before an execution request is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from fisheye.analysis.bout_kinematics_schema import (
    build_bout_kinematics_array_declarations,
)
from fisheye.analysis.detection_occupancy_schema import (
    build_occupancy_array_declarations,
)
from fisheye.analysis.swim_bout_schema import build_swim_bout_array_declarations
from fisheye.shared.zarr.analysis_benchmark_suite import (
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes


@dataclass(frozen=True)
class _ArrayView:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]


class _GroupView:
    def __init__(self) -> None:
        self._arrays: dict[str, _ArrayView] = {}
        self._groups: dict[str, _GroupView] = {}

    def add(self, path: str, array: _ArrayView) -> None:
        parts = path.split("/")
        current = self
        for part in parts[:-1]:
            current = current._groups.setdefault(part, _GroupView())
        if parts[-1] in current._arrays or parts[-1] in current._groups:
            raise ValueError(f"duplicate suite array path {path!r}")
        current._arrays[parts[-1]] = array

    def arrays(self):
        return tuple(self._arrays.items())

    def groups(self):
        return tuple(self._groups.items())


def _exact_tabular_builder(
    stage_id: str,
) -> Callable[[_GroupView], tuple[Any, ...]]:
    builders: dict[str, Callable[[_GroupView], tuple[Any, ...]]] = {
        "swim_bouts": lambda group: build_swim_bout_array_declarations(
            group, byte_planner_adopted=True
        ),
        "bout_kinematics": lambda group: build_bout_kinematics_array_declarations(
            group, byte_planner_adopted=True
        ),
        "detection_occupancy": lambda group: build_occupancy_array_declarations(
            group, session=False, byte_planner_adopted=True
        ),
        "session_occupancy": lambda group: build_occupancy_array_declarations(
            group, session=True, byte_planner_adopted=True
        ),
    }
    try:
        return builders[stage_id]
    except KeyError as exc:
        raise ValueError(
            f"stage {stage_id!r} has no exact-tabular suite validator"
        ) from exc


def require_exact_tabular_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Require the suite plan to equal the live exact family declaration set."""

    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if payload["family_id"] != stage_id:
        raise ValueError("benchmark suite family differs from the requested stage")
    records = payload["storage_plan_receipt"]["payload"]["arrays"]
    if not isinstance(records, list) or not records:
        raise ValueError("benchmark suite must plan one nonempty family array set")

    group = _GroupView()
    observed_declarations: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("benchmark suite contains a non-object array record")
        facts = record.get("observed_facts")
        declaration = record.get("declaration")
        if not isinstance(facts, Mapping) or not isinstance(declaration, Mapping):
            raise ValueError("benchmark suite array lacks facts or declaration")
        path = facts.get("path")
        shape = facts.get("shape")
        dtype = facts.get("dtype")
        if type(path) is not str or not isinstance(shape, list) or type(dtype) is not str:
            raise ValueError("benchmark suite array facts are not canonical")
        group.add(
            path,
            _ArrayView(
                shape=tuple(int(value) for value in shape),
                dtype=np.dtype(dtype),
            ),
        )
        observed_declarations.append(declaration)

    expected = [
        declaration.as_manifest()
        for declaration in _exact_tabular_builder(stage_id)(group)
    ]
    if canonical_json_bytes(observed_declarations) != canonical_json_bytes(expected):
        raise ValueError(
            "benchmark suite declarations differ from the live family projection"
        )


__all__ = ["require_exact_tabular_execution_suite"]
