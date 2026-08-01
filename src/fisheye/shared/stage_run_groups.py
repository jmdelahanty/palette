"""Canonical stage-to-run-parent mapping for Palette Zarr stores."""

from __future__ import annotations

from collections.abc import Sequence


STAGE_RUN_PARENTS: dict[str, tuple[str, ...]] = {
    # Live background writers use background_runs/, while some legacy stores and
    # catalog entries still name the artifact background/.
    "background": ("background_runs", "background"),
    "detect": ("detect_runs",),
    "detect_quality": ("detect_runs",),
    "refined_detect": ("refined_detect_runs", "refined_runs"),
    "refine": ("refined_detect_runs", "refined_runs"),
    "crop": ("crop_runs",),
    "keypoints": ("keypoints_runs",),
    "refined_keypoints": ("refined_keypoints_runs", "keypoints_refined_runs"),
    "keypoints_refine": ("refined_keypoints_runs", "keypoints_refined_runs"),
    # Deprecated stage families remain mapped so historical stores and registry
    # rows can still be read; planning/catalog code decides whether to recommend
    # them.
    "eye_masks": ("eye_masks_runs",),
    "refined_eye_masks": ("refined_eye_masks_runs",),
    "subject_masks": ("subject_mask_runs",),
    "refined_subject_masks": ("refined_subject_masks_runs",),
    "arena_assignment": ("arena_assignment_runs",),
    "assign_ids": ("arena_assignment_runs",),
    "track": ("tracking_runs",),
    "tracks": ("tracking_runs",),
    "tracking": ("tracking_runs",),
    # Derived-analysis families live below analysis/.  Qualified track run
    # names retain their online/offline scope (for example
    # ``offline/track_001``) beneath this parent.
    "track_kinematics": ("analysis/track_kinematics_runs",),
    "swim_bouts": ("analysis/swim_bout_runs",),
    "bout_kinematics": ("analysis/bout_kinematics_runs",),
    "eye_angles": ("analysis/eye_angle_runs",),
    "subject_shape": ("analysis/subject_shape_runs",),
    "tail_kinematics": ("analysis/tail_kinematics_runs",),
    "stimulus_response": ("analysis/stimulus_response_runs",),
}


def stage_run_parent_paths(
    stage_id: str,
    *,
    artifact_families: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return candidate Zarr run-parent paths for a stage id."""

    stage = str(stage_id).strip()
    mapped = STAGE_RUN_PARENTS.get(stage)
    if mapped:
        return mapped
    return tuple(str(value).strip("/") for value in artifact_families if str(value).strip("/"))
