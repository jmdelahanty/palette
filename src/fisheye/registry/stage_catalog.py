"""Canonical recording-stage vocabulary for registry/status surfaces.

This module is intentionally declarative. It does not change runtime pipeline
behavior yet; it gives pipeline, launcher, registry, and status-page code one
place to converge on stable stage identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass


CORE_PIPELINE = "core_pipeline"
RECORDING_METADATA = "recording_metadata"
DERIVED_ANALYSIS = "derived_analysis"
TUNING = "tuning"


@dataclass(frozen=True)
class StageSpec:
    """Stable description of a registry/status stage.

    ``id`` is the canonical registry/status name. ``aliases`` are legacy or UI
    names that may be translated to the canonical id. ``artifact_families`` are
    advisory Zarr families; they are not exhaustive path contracts.
    """

    id: str
    aliases: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    artifact_families: tuple[str, ...] = ()
    category: str = CORE_PIPELINE
    description: str = ""
    deprecated: bool = False


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        id="raw",
        aliases=("import",),
        invalidates=("downsample", "background"),
        category=CORE_PIPELINE,
        description="Raw recording imported or registered for analysis.",
    ),
    StageSpec(
        id="downsample",
        depends_on=("raw",),
        invalidates=("background",),
        artifact_families=("raw_video/images_ds",),
        category=CORE_PIPELINE,
        description="Downsampled video array derived from raw frames.",
    ),
    StageSpec(
        id="calibration",
        category=RECORDING_METADATA,
        description="Camera/stimulus calibration metadata available for the recording.",
    ),
    StageSpec(
        id="stimulus",
        artifact_families=("analysis/stimulus_runs",),
        category=RECORDING_METADATA,
        description="Stimulus timing and canonical step metadata available for the recording.",
    ),
    StageSpec(
        id="background",
        depends_on=("raw",),
        invalidates=("detect",),
        artifact_families=("background",),
        description="Background model used by traditional detection stages.",
    ),
    StageSpec(
        id="detect",
        invalidates=("detect_quality",),
        artifact_families=("detect_runs",),
        description="Raw detection/model output. Immutable except explicit overwrite.",
    ),
    StageSpec(
        id="detect_quality",
        depends_on=("detect",),
        invalidates=("refined_detect",),
        artifact_families=("detect_quality_runs",),
        description="Quality-control summaries for raw detections.",
    ),
    StageSpec(
        id="refined_detect",
        aliases=("refine",),
        depends_on=("detect_quality",),
        invalidates=("crop",),
        artifact_families=("refined_detect_runs",),
        description="Curated detection instances used as the authoring surface.",
    ),
    StageSpec(
        id="crop",
        depends_on=("refined_detect",),
        invalidates=("keypoints", "subject_masks"),
        artifact_families=("crop_runs",),
        description="ROI crops derived from refined detection instances.",
    ),
    StageSpec(
        id="keypoints",
        depends_on=("crop",),
        invalidates=("keypoint_quality",),
        artifact_families=("keypoints_runs",),
        description="Raw anatomical keypoint predictions.",
    ),
    StageSpec(
        id="keypoint_quality",
        depends_on=("keypoints",),
        invalidates=("refined_keypoints",),
        artifact_families=("keypoint_quality_runs",),
        description=(
            "Immutable source-bound keypoint diagnostics and usability proposals."
        ),
    ),
    StageSpec(
        id="refined_keypoints",
        aliases=("keypoints_refine",),
        depends_on=("keypoint_quality",),
        invalidates=("arena_assignment", "track_kinematics"),
        artifact_families=("refined_keypoints_runs",),
        description="Curated keypoint instances used downstream.",
    ),
    StageSpec(
        id="eye_masks",
        depends_on=("refined_keypoints",),
        invalidates=("refined_eye_masks",),
        artifact_families=("eye_masks_runs",),
        description="Raw eye mask predictions.",
        deprecated=True,
    ),
    StageSpec(
        id="refined_eye_masks",
        depends_on=("eye_masks",),
        artifact_families=("refined_eye_masks_runs",),
        description="Curated eye mask instances.",
        deprecated=True,
    ),
    StageSpec(
        id="subject_masks",
        depends_on=("crop",),
        invalidates=("refined_subject_masks",),
        artifact_families=("subject_mask_runs",),
        description="Raw whole-subject mask predictions.",
    ),
    StageSpec(
        id="refined_subject_masks",
        depends_on=("subject_masks",),
        invalidates=("subject_shape", "eye_angles"),
        artifact_families=("refined_subject_masks_runs",),
        description="Curated whole-subject mask instances.",
    ),
    StageSpec(
        id="arena_assignment",
        aliases=("assign_ids",),
        depends_on=("refined_detect",),
        invalidates=("tracks",),
        artifact_families=("arena_assignment_runs",),
        description="Assignment of detected subjects to arena identities.",
    ),
    StageSpec(
        id="tracks",
        aliases=("track",),
        depends_on=("arena_assignment",),
        artifact_families=("tracks",),
        description="Track-level subject identity output.",
    ),
    StageSpec(
        id="track_kinematics",
        depends_on=("refined_keypoints",),
        invalidates=(
            "swim_bouts",
            "track_kinematics_visualization",
            "stimulus_response",
            "bout_classification",
        ),
        artifact_families=("analysis/track_kinematics_runs",),
        category=DERIVED_ANALYSIS,
        description="Speed, distance, heading, and derived track kinematics.",
    ),
    StageSpec(
        id="swim_bouts",
        depends_on=("track_kinematics",),
        invalidates=(
            "track_kinematics_visualization",
            "bout_kinematics",
            "stimulus_response",
            "bout_classification",
        ),
        artifact_families=("analysis/swim_bout_runs",),
        category=DERIVED_ANALYSIS,
        description="Swim-bout segmentation candidates derived from kinematics.",
    ),
    StageSpec(
        id="track_kinematics_visualization",
        depends_on=("track_kinematics", "swim_bouts"),
        artifact_families=(
            "analysis/track_kinematics_visualization_runs/"
            "*/*/tracks/id_*/*/visualizations",
        ),
        category=DERIVED_ANALYSIS,
        description=(
            "Immutable interactive core-behavior render bound to a sealed "
            "track-motion run."
        ),
    ),
    StageSpec(
        id="bout_kinematics",
        depends_on=("swim_bouts", "eye_angles"),
        artifact_families=("analysis/bout_kinematics_runs",),
        category=DERIVED_ANALYSIS,
        description="Per-bout movement, heading, and eye-gaze metrics.",
    ),
    StageSpec(
        id="eye_angles",
        depends_on=("subject_shape",),
        invalidates=("bout_kinematics", "stimulus_response"),
        artifact_families=("analysis/eye_angle_runs",),
        category=DERIVED_ANALYSIS,
        description=(
            "Per-eye angle and vergence measurements from subject-shape eye "
            "geometry and its sealed canonical base-keypoint authority."
        ),
    ),
    StageSpec(
        id="subject_shape",
        depends_on=("refined_subject_masks",),
        invalidates=("eye_angles", "tail_kinematics", "tail_posture_view"),
        artifact_families=("analysis/subject_shape_runs",),
        category=DERIVED_ANALYSIS,
        description="Mask-derived body frame, centerline, and shape geometry.",
    ),
    StageSpec(
        id="tail_kinematics",
        depends_on=("subject_shape",),
        invalidates=("tail_posture_view",),
        artifact_families=("analysis/tail_kinematics_runs",),
        category=DERIVED_ANALYSIS,
        description="Frame-level tail angles, deflections, and curvature summaries.",
    ),
    StageSpec(
        id="tail_posture_view",
        depends_on=("subject_shape",),
        invalidates=("bout_classification",),
        artifact_families=("analysis/tail_posture_view_runs",),
        category=DERIVED_ANALYSIS,
        description="Tool-compatible tail posture views derived from subject shape.",
    ),
    StageSpec(
        id="bout_classification",
        depends_on=("tail_posture_view", "track_kinematics", "swim_bouts"),
        artifact_families=("analysis/bout_classification_runs",),
        category=DERIVED_ANALYSIS,
        description="Per-bout classifier labels and scores from Palette or external tools.",
    ),
    StageSpec(
        id="stimulus_response",
        depends_on=("stimulus", "track_kinematics", "swim_bouts"),
        artifact_families=("analysis/stimulus_response_runs",),
        category=DERIVED_ANALYSIS,
        description="Stimulus-aligned response metrics and summaries.",
    ),
    StageSpec(
        id="dish_mask",
        aliases=("mask",),
        category=TUNING,
        description="Dish/arena mask tuning metadata.",
    ),
    StageSpec(
        id="detection_tuning",
        aliases=("threshold",),
        category=TUNING,
        description="Detection threshold/tuning metadata.",
    ),
    StageSpec(
        id="keypoint_tuning",
        category=TUNING,
        description="Keypoint detector tuning metadata.",
    ),
    StageSpec(
        id="subject_mask_tuning",
        aliases=("subject-mask", "subject_mask"),
        category=TUNING,
        description="Subject mask tuner metadata.",
    ),
    StageSpec(
        id="eye_mask_tuning",
        aliases=("eye-mask", "eye_mask"),
        category=TUNING,
        description="Eye mask tuner metadata.",
        deprecated=True,
    ),
    StageSpec(
        id="subdish_mask_tuning",
        aliases=("subdish", "subdish-mask", "subdish_mask"),
        category=TUNING,
        description="Sub-dish/arena subdivision tuning metadata.",
    ),
)


STAGE_BY_ID: dict[str, StageSpec] = {spec.id: spec for spec in STAGE_SPECS}
ALIAS_TO_STAGE_ID: dict[str, str] = {
    alias: spec.id for spec in STAGE_SPECS for alias in spec.aliases
}
STAGE_IDS: tuple[str, ...] = tuple(spec.id for spec in STAGE_SPECS)
RECORDING_TUNING_STAGE_IDS: tuple[str, ...] = tuple(
    spec.id for spec in STAGE_SPECS if spec.category == TUNING
)
RECORDING_STATUS_STAGE_IDS: tuple[str, ...] = (
    "raw",
    "background",
    "detect",
    "detect_quality",
    "refined_detect",
    "crop",
    "keypoints",
    "refined_keypoints",
    "eye_masks",
    "refined_eye_masks",
    "subject_masks",
    "refined_subject_masks",
    "arena_assignment",
    "tracks",
    "track_kinematics",
    "swim_bouts",
    "bout_kinematics",
    "eye_angles",
    "subject_shape",
    "tail_kinematics",
    "tail_posture_view",
    "bout_classification",
    "stimulus_response",
    "stimulus",
    "calibration",
    *RECORDING_TUNING_STAGE_IDS,
)
DEPRECATED_STAGE_IDS: tuple[str, ...] = tuple(
    spec.id for spec in STAGE_SPECS if spec.deprecated
)


def canonical_stage_id(stage_id_or_alias: str) -> str:
    """Return the canonical stage id for a canonical id or unambiguous alias."""

    if stage_id_or_alias in STAGE_BY_ID:
        return stage_id_or_alias
    try:
        return ALIAS_TO_STAGE_ID[stage_id_or_alias]
    except KeyError as exc:
        raise KeyError(f"unknown stage id or alias: {stage_id_or_alias!r}") from exc


def get_stage_spec(stage_id_or_alias: str) -> StageSpec:
    """Return a :class:`StageSpec` by canonical id or alias."""

    return STAGE_BY_ID[canonical_stage_id(stage_id_or_alias)]


def dependency_map() -> dict[str, tuple[str, ...]]:
    """Return canonical stage dependencies keyed by canonical stage id."""

    return {spec.id: spec.depends_on for spec in STAGE_SPECS}


def invalidation_map() -> dict[str, tuple[str, ...]]:
    """Return direct downstream invalidations keyed by canonical stage id."""

    return {spec.id: spec.invalidates for spec in STAGE_SPECS}


def artifact_family_map() -> dict[str, tuple[str, ...]]:
    """Return advisory artifact families keyed by canonical stage id."""

    return {spec.id: spec.artifact_families for spec in STAGE_SPECS}


def deprecated_stage_ids() -> tuple[str, ...]:
    """Return canonical stage ids retained only for historical registry rows."""

    return DEPRECATED_STAGE_IDS


def recording_status_stage_ids() -> tuple[str, ...]:
    """Return canonical recording-status stage ids in display/backfill order."""

    return RECORDING_STATUS_STAGE_IDS


def recording_tuning_stage_ids() -> tuple[str, ...]:
    """Return canonical tuning stage ids in display/backfill order."""

    return RECORDING_TUNING_STAGE_IDS


__all__ = [
    "ALIAS_TO_STAGE_ID",
    "CORE_PIPELINE",
    "DERIVED_ANALYSIS",
    "RECORDING_METADATA",
    "RECORDING_STATUS_STAGE_IDS",
    "RECORDING_TUNING_STAGE_IDS",
    "DEPRECATED_STAGE_IDS",
    "STAGE_BY_ID",
    "STAGE_IDS",
    "STAGE_SPECS",
    "TUNING",
    "StageSpec",
    "artifact_family_map",
    "canonical_stage_id",
    "deprecated_stage_ids",
    "dependency_map",
    "get_stage_spec",
    "invalidation_map",
    "recording_status_stage_ids",
    "recording_tuning_stage_ids",
]
