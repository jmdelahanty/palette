"""Node-local materializers for large recording analysis products."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "TailKinematicsMaterializationPlan",
    "TrackKinematicsMaterializationPlan",
    "build_tail_kinematics_materialization_plan",
    "build_track_kinematics_materialization_plan",
    "materialize_tail_kinematics",
    "materialize_track_kinematics",
]


def __getattr__(name: str) -> Any:
    """Load public materializer APIs lazily so ``python -m`` stays warning-free."""

    if name not in __all__:
        raise AttributeError(name)
    module_name = (
        ".track_kinematics"
        if name
        in {
            "TrackKinematicsMaterializationPlan",
            "build_track_kinematics_materialization_plan",
            "materialize_track_kinematics",
        }
        else ".tail_kinematics"
    )
    module = import_module(module_name, __name__)
    return getattr(module, name)
