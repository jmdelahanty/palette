"""Lightweight renderer contract for persisted bout-kinematics plots."""

from __future__ import annotations

from typing import Optional


BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID = "palette.plot_spec.bout_kinematics_summary.v1"
BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID = "palette.plot_spec.bout_movement_summary.v1"
BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID = "palette.plot_spec.bout_eye_gaze_summary.v1"

BOUT_HEADING_PLOT_RENDERER = "palette-bout-kinematics-heading-v1"
BOUT_MOVEMENT_PLOT_RENDERER = "palette-bout-kinematics-movement-v1"
BOUT_EYE_GAZE_PLOT_RENDERER = "palette-bout-kinematics-eye-gaze-v1"
LEGACY_BOUT_PLOT_RENDERER = "matplotlib_static_plotly_spec.v1"

BOUT_KINEMATICS_PNG_PREFIX = "bout_kinematics_summary"
BOUT_MOVEMENT_PNG_PREFIX = "bout_movement_summary"
BOUT_EYE_GAZE_PNG_PREFIX = "bout_eye_gaze_summary"

BOUT_SCHEMA_TO_RENDERER = {
    BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID: BOUT_HEADING_PLOT_RENDERER,
    BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID: BOUT_MOVEMENT_PLOT_RENDERER,
    BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID: BOUT_EYE_GAZE_PLOT_RENDERER,
}
BOUT_SCHEMA_TO_ANALYSIS_ID = {
    BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID: "heading",
    BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID: "movement",
    BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID: "eye_gaze",
}
BOUT_ANALYSIS_ID_TO_SCHEMA = {
    analysis_id: schema_id for schema_id, analysis_id in BOUT_SCHEMA_TO_ANALYSIS_ID.items()
}
BOUT_ANALYSIS_ID_TO_ARTIFACT_PREFIX = {
    "heading": BOUT_KINEMATICS_PNG_PREFIX,
    "movement": BOUT_MOVEMENT_PNG_PREFIX,
    "eye_gaze": BOUT_EYE_GAZE_PNG_PREFIX,
}
BOUT_PLOT_RENDERERS = tuple(BOUT_SCHEMA_TO_RENDERER.values())
BOUT_PLOT_SPEC_SCHEMA_IDS = tuple(BOUT_SCHEMA_TO_RENDERER)


def effective_bout_renderer(renderer: object, schema_id: object) -> str:
    """Return a dedicated renderer for an exact legacy bout schema.

    The historical renderer ID was shared by unrelated Matplotlib-backed plot
    specs. It is therefore recognized only when paired with one of the three
    bout schema IDs above.
    """

    persisted = str(renderer or "").strip()
    schema = str(schema_id or "").strip()
    if persisted == LEGACY_BOUT_PLOT_RENDERER and schema in BOUT_SCHEMA_TO_RENDERER:
        return BOUT_SCHEMA_TO_RENDERER[schema]
    return persisted


def bout_analysis_id_for_schema(schema_id: object) -> Optional[str]:
    return BOUT_SCHEMA_TO_ANALYSIS_ID.get(str(schema_id or "").strip())


def bout_schema_for_artifact_name(artifact_name: object) -> Optional[str]:
    name = str(artifact_name or "").strip()
    for analysis_id, prefix in BOUT_ANALYSIS_ID_TO_ARTIFACT_PREFIX.items():
        if name.startswith(f"{prefix}_track_") and name.endswith("_interactive"):
            return BOUT_ANALYSIS_ID_TO_SCHEMA[analysis_id]
    return None
