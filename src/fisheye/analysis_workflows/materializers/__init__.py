"""Node-local materializers for large recording analysis products."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BoutKinematicsComputePlan",
    "BoutKinematicsStoragePlan",
    "EyeAngleMaterializationPlan",
    "StimulusResponseMaterializationPlan",
    "SwimBoutMaterializationPlan",
    "TailKinematicsMaterializationPlan",
    "TrackKinematicsMaterializationPlan",
    "build_bout_kinematics_compute_plan",
    "build_bout_kinematics_storage_plan",
    "build_eye_angle_materialization_admission_receipt",
    "build_eye_angle_materialization_plan",
    "apply_eye_angle_materialization_plan",
    "build_stimulus_response_materialization_plan",
    "build_swim_bout_materialization_plan",
    "build_tail_kinematics_materialization_plan",
    "build_track_kinematics_materialization_plan",
    "materialize_bout_kinematics_compute",
    "materialize_bout_kinematics_storage",
    "promote_bout_kinematics_candidate",
    "load_eye_angle_materialization_admission_receipt",
    "materialize_eye_angles",
    "materialize_stimulus_response",
    "materialize_swim_bouts",
    "materialize_tail_kinematics",
    "materialize_track_kinematics",
    "validate_eye_angle_materialization_admission_receipt",
    "write_eye_angle_materialization_admission_receipt",
]

_MODULE_BY_NAME = {
    "BoutKinematicsComputePlan": ".bout_kinematics",
    "BoutKinematicsStoragePlan": ".bout_kinematics",
    "build_bout_kinematics_compute_plan": ".bout_kinematics",
    "build_bout_kinematics_storage_plan": ".bout_kinematics",
    "materialize_bout_kinematics_compute": ".bout_kinematics",
    "materialize_bout_kinematics_storage": ".bout_kinematics",
    "promote_bout_kinematics_candidate": ".bout_kinematics",
    "EyeAngleMaterializationPlan": ".eye_angles",
    "build_eye_angle_materialization_admission_receipt": ".eye_angles",
    "build_eye_angle_materialization_plan": ".eye_angles",
    "apply_eye_angle_materialization_plan": ".eye_angles",
    "load_eye_angle_materialization_admission_receipt": ".eye_angles",
    "materialize_eye_angles": ".eye_angles",
    "validate_eye_angle_materialization_admission_receipt": ".eye_angles",
    "write_eye_angle_materialization_admission_receipt": ".eye_angles",
    "StimulusResponseMaterializationPlan": ".stimulus_response",
    "build_stimulus_response_materialization_plan": ".stimulus_response",
    "materialize_stimulus_response": ".stimulus_response",
    "SwimBoutMaterializationPlan": ".swim_bouts",
    "build_swim_bout_materialization_plan": ".swim_bouts",
    "materialize_swim_bouts": ".swim_bouts",
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
