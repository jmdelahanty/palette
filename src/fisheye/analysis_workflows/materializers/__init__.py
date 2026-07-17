"""Node-local materializers for large recording analysis products."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BoutKinematicsStoragePlan",
    "EyeAngleMaterializationPlan",
    "TailKinematicsMaterializationPlan",
    "TrackKinematicsMaterializationPlan",
    "build_bout_kinematics_storage_plan",
    "build_eye_angle_materialization_plan",
    "build_tail_kinematics_materialization_plan",
    "build_track_kinematics_materialization_plan",
    "materialize_bout_kinematics_storage",
    "materialize_eye_angles",
    "materialize_tail_kinematics",
    "materialize_track_kinematics",
]

_MODULE_BY_NAME = {
    "BoutKinematicsStoragePlan": ".bout_kinematics",
    "build_bout_kinematics_storage_plan": ".bout_kinematics",
    "materialize_bout_kinematics_storage": ".bout_kinematics",
    "EyeAngleMaterializationPlan": ".eye_angles",
    "build_eye_angle_materialization_plan": ".eye_angles",
    "materialize_eye_angles": ".eye_angles",
    "TrackKinematicsMaterializationPlan": ".track_kinematics",
    "build_track_kinematics_materialization_plan": ".track_kinematics",
    "materialize_track_kinematics": ".track_kinematics",
    "TailKinematicsMaterializationPlan": ".tail_kinematics",
    "build_tail_kinematics_materialization_plan": ".tail_kinematics",
    "materialize_tail_kinematics": ".tail_kinematics",
}


def __getattr__(name: str) -> Any:
    """Load public materializer APIs lazily so ``python -m`` stays warning-free."""

    if name not in __all__:
        raise AttributeError(name)
    module_name = _MODULE_BY_NAME[name]
    module = import_module(module_name, __name__)
    return getattr(module, name)
