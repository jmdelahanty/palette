"""Static in-library catalog for core and stimulus report providers."""

from __future__ import annotations

from fisheye.visualization.bout_kinematics_interactive import (
    BOUT_HEADING_PLOT_RENDERER,
    BOUT_MOVEMENT_PLOT_RENDERER,
)

from .models import (
    AnalysisFamilySpec,
    ArtifactSelector,
    EntityScope,
    ProviderSpec,
    SourceRequirement,
    VisualizationSpec,
)


ANALYSIS_FAMILIES: dict[str, AnalysisFamilySpec] = {
    spec.family_id: spec
    for spec in (
        AnalysisFamilySpec(
            family_id="stimulus.metadata",
            label="Canonical stimulus metadata",
            stage_id="stimulus",
            run_parent_paths=("analysis/stimulus_runs",),
        ),
        AnalysisFamilySpec(
            family_id="core.track_kinematics",
            label="Track kinematics",
            stage_id="track_kinematics",
            run_parent_paths=(
                "analysis/track_kinematics_runs/offline",
                "analysis/track_kinematics_runs/online",
            ),
            source_requirements=(
                SourceRequirement(
                    source_id="refined_keypoints",
                    any_group_paths=("refined_keypoints_runs",),
                ),
            ),
        ),
        AnalysisFamilySpec(
            family_id="core.swim_bouts",
            label="Swim bouts",
            stage_id="swim_bouts",
            run_parent_paths=("analysis/swim_bout_runs",),
            prerequisites=("core.track_kinematics",),
            entity_id_attrs=("track_id", "source_track_id"),
        ),
        AnalysisFamilySpec(
            family_id="core.bout_kinematics",
            label="Bout kinematics",
            stage_id="bout_kinematics",
            run_parent_paths=("analysis/bout_kinematics_runs",),
            prerequisites=("core.track_kinematics", "core.swim_bouts"),
            entity_id_attrs=("source_track_id", "track_id"),
        ),
        AnalysisFamilySpec(
            family_id="core.eye_angles",
            label="Eye angles",
            stage_id="eye_angles",
            run_parent_paths=("analysis/eye_angle_runs",),
            source_requirements=(
                SourceRequirement(
                    source_id="refined_keypoints",
                    any_group_paths=("refined_keypoints_runs",),
                ),
                SourceRequirement(
                    source_id="refined_subject_masks",
                    any_group_paths=("refined_subject_masks_runs",),
                ),
            ),
        ),
        AnalysisFamilySpec(
            family_id="core.session_occupancy",
            label="Full-session occupancy",
            stage_id="detection_occupancy",
            run_parent_paths=("analysis/session_occupancy_runs",),
            source_requirements=(
                SourceRequirement(
                    source_id="detections",
                    any_group_paths=("refined_detect_runs", "detect_runs"),
                ),
            ),
        ),
        AnalysisFamilySpec(
            family_id="stimulus.chaser_distance",
            label="Chaser distance",
            stage_id="chaser_distance",
            run_parent_paths=("analysis/chaser_distance_runs",),
            prerequisites=("stimulus.metadata", "core.track_kinematics"),
        ),
        AnalysisFamilySpec(
            family_id="stimulus.response",
            label="Stimulus response",
            stage_id="stimulus_response",
            run_parent_paths=("analysis/stimulus_response_runs",),
            prerequisites=(
                "stimulus.metadata",
                "core.track_kinematics",
                "core.swim_bouts",
            ),
        ),
    )
}


VISUALIZATIONS: dict[str, VisualizationSpec] = {
    spec.visualization_id: spec
    for spec in (
        VisualizationSpec(
            visualization_id="core.track_kinematics.overview",
            label="Track kinematics overview",
            provider_id="core_behavior.v1",
            analysis_family_id="core.track_kinematics",
            selector=ArtifactSelector(
                "visualizations/track_kinematics_summary_track_{entity_id}_png"
            ),
            entity_scope=EntityScope.TRACK,
            entity_source_family_id="core.track_kinematics",
            visualization_contract_id="palette.core.track_kinematics.summary.v1",
            renderer="palette-track-kinematics-summary-v1",
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="core.position.xy_trace",
            label="X/Y position traces",
            provider_id="core_behavior.v1",
            analysis_family_id="core.track_kinematics",
            selector=ArtifactSelector(
                "visualizations/position_xy_trace_track_{entity_id}_png"
            ),
            entity_scope=EntityScope.TRACK,
            entity_source_family_id="core.track_kinematics",
            visualization_contract_id="palette.core.position.xy_trace.v1",
            renderer="palette-core-position-xy-trace-v1",
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="core.swim_bouts.overview",
            label="Swim-bout summary",
            provider_id="core_behavior.v1",
            analysis_family_id="core.swim_bouts",
            selector=ArtifactSelector("visualizations/swim_bout_summary_png"),
            entity_scope=EntityScope.TRACK,
            entity_source_family_id="core.track_kinematics",
            visualization_contract_id="palette.core.swim_bouts.summary.v1",
            renderer="palette-core-swim-bouts-summary-v1",
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="core.bout_kinematics.movement",
            label="Bout movement summary",
            provider_id="core_behavior.v1",
            analysis_family_id="core.bout_kinematics",
            selector=ArtifactSelector(
                "visualizations/bout_movement_summary_track_{entity_id}_png"
            ),
            entity_scope=EntityScope.TRACK,
            entity_source_family_id="core.track_kinematics",
            visualization_contract_id="palette.core.bout_kinematics.movement.v1",
            renderer=BOUT_MOVEMENT_PLOT_RENDERER,
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="core.bout_kinematics.heading",
            label="Bout heading summary",
            provider_id="core_behavior.v1",
            analysis_family_id="core.bout_kinematics",
            selector=ArtifactSelector(
                "visualizations/bout_kinematics_summary_track_{entity_id}_png"
            ),
            entity_scope=EntityScope.TRACK,
            entity_source_family_id="core.track_kinematics",
            visualization_contract_id="palette.core.bout_kinematics.heading.v1",
            renderer=BOUT_HEADING_PLOT_RENDERER,
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="core.eye_angles.overview",
            label="Eye-angle and convergence summary",
            provider_id="core_behavior.v1",
            analysis_family_id="core.eye_angles",
            selector=ArtifactSelector("visualizations/eye_angle_dashboard_eye_frame_png"),
            visualization_contract_id="palette.core.eye_angles.summary.v1",
            renderer="palette-eye-angle-dashboard-v1",
            renderer_version="1",
            required_by_default=False,
        ),
        VisualizationSpec(
            visualization_id="core.position.occupancy",
            label="Full-session position occupancy",
            provider_id="core_behavior.v1",
            analysis_family_id="core.session_occupancy",
            selector=ArtifactSelector("visualizations/session_occupancy_overview_png"),
            visualization_contract_id="palette.core.position.session_occupancy.v1",
            renderer="fisheye.analysis.detection_occupancy_runs",
            renderer_version="1",
            required_by_default=False,
        ),
        VisualizationSpec(
            visualization_id="stimulus.chaser.distance_trace",
            label="Chaser distance trace",
            provider_id="stimulus.chaser.v1",
            analysis_family_id="stimulus.chaser_distance",
            selector=ArtifactSelector("visualizations/chaser_distance_timeseries_png"),
            visualization_contract_id="palette.stimulus.chaser.distance_trace.v1",
            renderer="palette-chaser-distance-timeseries-v1",
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="stimulus.chaser.distance_distribution",
            label="Chaser distance distributions",
            provider_id="stimulus.chaser.v1",
            analysis_family_id="stimulus.chaser_distance",
            selector=ArtifactSelector(
                "visualizations/chaser_distance_epoch_distribution_png"
            ),
            visualization_contract_id=(
                "palette.stimulus.chaser.distance_epoch_distribution.v1"
            ),
            renderer="palette-chaser-distance-epoch-distribution-v1",
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="stimulus.chaser.egocentric_bearing",
            label="Egocentric chaser bearing",
            provider_id="stimulus.chaser.v1",
            analysis_family_id="stimulus.chaser_distance",
            selector=ArtifactSelector(
                "egocentric_bearing/*/visualizations/"
                "egocentric_bearing_pre_post_polar_png"
            ),
            visualization_contract_id=(
                "palette.chaser_egocentric_bearing.pre_post_polar_density.v2"
            ),
            renderer="fisheye.analysis.chaser_egocentric_bearing",
            renderer_version="2",
        ),
        VisualizationSpec(
            visualization_id="stimulus.moving_grating.omr_summary",
            label="Moving-grating OMR summary",
            provider_id="stimulus.moving_grating.v1",
            analysis_family_id="stimulus.response",
            selector=ArtifactSelector("visualizations/stimulus_response_omr_summary_png"),
            visualization_contract_id="palette.stimulus.moving_grating.omr_summary.v1",
            renderer="palette-stimulus-response-omr-summary-v1",
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="stimulus.moving_grating.bout_trajectory",
            label="Moving-grating bout trajectories",
            provider_id="stimulus.moving_grating.v1",
            analysis_family_id="stimulus.response",
            selector=ArtifactSelector(
                "visualizations/stimulus_response_omr_bout_trajectory_png"
            ),
            visualization_contract_id=(
                "palette.stimulus.moving_grating.omr_bout_trajectory.v1"
            ),
            renderer="palette-stimulus-response-omr-summary-v1",
            renderer_version="1",
            required_by_default=False,
        ),
        VisualizationSpec(
            visualization_id="stimulus.concentric_grating.response_summary",
            label="Concentric-grating response summary",
            provider_id="stimulus.concentric_grating.v1",
            analysis_family_id="stimulus.response",
            selector=ArtifactSelector(
                "visualizations/stimulus_response_concentric_summary_png"
            ),
            visualization_contract_id=(
                "palette.stimulus.concentric_grating.response_summary.v1"
            ),
            renderer="palette-stimulus-response-concentric-summary-v1",
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="stimulus.looming.response_summary",
            label="Looming response summary",
            provider_id="stimulus.looming.v1",
            analysis_family_id="stimulus.response",
            selector=ArtifactSelector(
                "visualizations/stimulus_response_looming_summary_png"
            ),
            visualization_contract_id="palette.stimulus.looming.response_summary.v1",
            renderer="palette-stimulus-response-looming-summary-v1",
            renderer_version="1",
        ),
        VisualizationSpec(
            visualization_id="stimulus.flash.response_summary",
            label="Flash response summary",
            provider_id="stimulus.flash.v1",
            analysis_family_id="stimulus.response",
            selector=ArtifactSelector(
                "visualizations/stimulus_response_flash_summary_png"
            ),
            visualization_contract_id="palette.stimulus.flash.response_summary.v1",
            renderer="palette-stimulus-response-flash-summary-v1",
            renderer_version="1",
        ),
    )
}


PROVIDERS: dict[str, ProviderSpec] = {
    spec.provider_id: spec
    for spec in (
        ProviderSpec(
            provider_id="core_behavior.v1",
            label="Core behavior",
            always_applicable=True,
            visualization_ids=tuple(
                spec.visualization_id
                for spec in VISUALIZATIONS.values()
                if spec.provider_id == "core_behavior.v1"
            ),
        ),
        ProviderSpec(
            provider_id="stimulus.chaser.v1",
            label="Chaser stimulus",
            stimulus_modes=("CHASER",),
            visualization_ids=tuple(
                spec.visualization_id
                for spec in VISUALIZATIONS.values()
                if spec.provider_id == "stimulus.chaser.v1"
            ),
        ),
        ProviderSpec(
            provider_id="stimulus.moving_grating.v1",
            label="Moving-grating stimulus",
            stimulus_modes=("MOVING_GRATING",),
            visualization_ids=tuple(
                spec.visualization_id
                for spec in VISUALIZATIONS.values()
                if spec.provider_id == "stimulus.moving_grating.v1"
            ),
        ),
        ProviderSpec(
            provider_id="stimulus.concentric_grating.v1",
            label="Concentric-grating stimulus",
            stimulus_modes=("CONCENTRIC_GRATING",),
            visualization_ids=tuple(
                spec.visualization_id
                for spec in VISUALIZATIONS.values()
                if spec.provider_id == "stimulus.concentric_grating.v1"
            ),
        ),
        ProviderSpec(
            provider_id="stimulus.looming.v1",
            label="Looming stimulus",
            stimulus_modes=("LOOMING_DOT",),
            visualization_ids=tuple(
                spec.visualization_id
                for spec in VISUALIZATIONS.values()
                if spec.provider_id == "stimulus.looming.v1"
            ),
        ),
        ProviderSpec(
            provider_id="stimulus.flash.v1",
            label="Flash stimulus",
            stimulus_modes=("DARK_FLASH", "BRIGHT_FLASH"),
            visualization_ids=tuple(
                spec.visualization_id
                for spec in VISUALIZATIONS.values()
                if spec.provider_id == "stimulus.flash.v1"
            ),
        ),
    )
}


def validate_catalog() -> None:
    """Fail fast when static declarations reference unknown catalog entries."""

    for visualization in VISUALIZATIONS.values():
        if visualization.provider_id not in PROVIDERS:
            raise ValueError(
                f"{visualization.visualization_id}: unknown provider "
                f"{visualization.provider_id!r}"
            )
        if visualization.analysis_family_id not in ANALYSIS_FAMILIES:
            raise ValueError(
                f"{visualization.visualization_id}: unknown analysis family "
                f"{visualization.analysis_family_id!r}"
            )
        if visualization.entity_source_family_id is not None:
            if visualization.entity_source_family_id not in ANALYSIS_FAMILIES:
                raise ValueError(
                    f"{visualization.visualization_id}: unknown entity source family "
                    f"{visualization.entity_source_family_id!r}"
                )
    for provider in PROVIDERS.values():
        for visualization_id in provider.visualization_ids:
            visualization = VISUALIZATIONS.get(visualization_id)
            if visualization is None:
                raise ValueError(
                    f"{provider.provider_id}: unknown visualization {visualization_id!r}"
                )
            if visualization.provider_id != provider.provider_id:
                raise ValueError(
                    f"{provider.provider_id}: visualization {visualization_id!r} belongs "
                    f"to {visualization.provider_id!r}"
                )


validate_catalog()
