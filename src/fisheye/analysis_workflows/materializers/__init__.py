"""Node-local materializers for large recording analysis products."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "TailKinematicsMaterializationPlan",
    "build_tail_kinematics_materialization_plan",
    "materialize_tail_kinematics",
]


def __getattr__(name: str) -> Any:
    """Load public materializer APIs lazily so ``python -m`` stays warning-free."""

    if name not in __all__:
        raise AttributeError(name)
    module = import_module(".tail_kinematics", __name__)
    return getattr(module, name)
