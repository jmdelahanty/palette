"""Curated visualization artifact profiles available to montage workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import MontageArtifactSpec


@dataclass(frozen=True)
class PlotProfile:
    profile_id: str
    label: str
    title: str
    artifact_path_template: str
    required_parameters: tuple[str, ...]
    default_max_image_height: int
    visualization_contract_id: str | None = None

    def artifact_spec(self, parameters: Mapping[str, str | None]) -> MontageArtifactSpec:
        missing = [name for name in self.required_parameters if not parameters.get(name)]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            raise ValueError(f"Plot type {self.profile_id!r} requires {flags}.")
        path = self.artifact_path_template.format(**parameters)
        return MontageArtifactSpec(
            artifact_id=self.profile_id,
            label=self.label,
            path=path,
            visualization_contract_id=self.visualization_contract_id,
        )


PLOT_PROFILES: dict[str, PlotProfile] = {
    profile.profile_id: profile
    for profile in (
        PlotProfile(
            profile_id="detection-occupancy",
            label="Detection occupancy",
            title="Detection occupancy",
            artifact_path_template=(
                "analysis/detection_occupancy_runs/{detection_occupancy_run}/"
                "visualizations/detection_occupancy_overview_png"
            ),
            required_parameters=("detection_occupancy_run",),
            default_max_image_height=420,
        ),
        PlotProfile(
            profile_id="chaser-distance-timeseries",
            label="Chaser distance timeseries",
            title="Fish-to-chaser distance timeseries",
            artifact_path_template=(
                "analysis/chaser_distance_runs/{chaser_distance_run}/"
                "visualizations/chaser_distance_timeseries_png"
            ),
            required_parameters=("chaser_distance_run",),
            default_max_image_height=360,
        ),
        PlotProfile(
            profile_id="chaser-distance-median",
            label="Chaser distance medians",
            title="Fish-to-chaser distance medians by epoch",
            artifact_path_template=(
                "analysis/chaser_distance_runs/{chaser_distance_run}/"
                "visualizations/chaser_distance_epoch_median_png"
            ),
            required_parameters=("chaser_distance_run",),
            default_max_image_height=380,
        ),
        PlotProfile(
            profile_id="chaser-distance-distribution",
            label="Chaser distance distributions",
            title="Fish-to-chaser distance distributions by epoch",
            artifact_path_template=(
                "analysis/chaser_distance_runs/{chaser_distance_run}/"
                "visualizations/chaser_distance_epoch_distribution_png"
            ),
            required_parameters=("chaser_distance_run",),
            default_max_image_height=280,
        ),
        PlotProfile(
            profile_id="egocentric-bearing-polar",
            label="Egocentric polar bearing",
            title="Pre/post egocentric polar bearing",
            artifact_path_template=(
                "analysis/chaser_distance_runs/{chaser_distance_run}/egocentric_bearing/"
                "{egocentric_component}/visualizations/egocentric_bearing_pre_post_polar_png"
            ),
            required_parameters=("chaser_distance_run", "egocentric_component"),
            default_max_image_height=540,
            visualization_contract_id=(
                "palette.chaser_egocentric_bearing.pre_post_polar_density.v2"
            ),
        ),
        PlotProfile(
            profile_id="egocentric-bearing-point-cloud",
            label="Egocentric bearing point cloud",
            title="Pre/post egocentric bearing point clouds",
            artifact_path_template=(
                "analysis/chaser_distance_runs/{chaser_distance_run}/egocentric_bearing/"
                "{egocentric_component}/visualizations/"
                "egocentric_bearing_pre_post_polar_point_cloud_png"
            ),
            required_parameters=("chaser_distance_run", "egocentric_component"),
            default_max_image_height=540,
            visualization_contract_id=(
                "palette.chaser_egocentric_bearing.pre_post_polar_point_cloud.v2"
            ),
        ),
        PlotProfile(
            profile_id="fish-escape-outcome-timeline",
            label="Fish escape outcome timeline",
            title="Successful versus failed fish escape timeline",
            artifact_path_template=(
                "analysis/chaser_distance_runs/{chaser_distance_run}/chaser_escape_freeze/"
                "{escape_freeze_component}/visualizations/escape_freeze_trial_outcome_timeline_png"
            ),
            required_parameters=("chaser_distance_run", "escape_freeze_component"),
            default_max_image_height=260,
        ),
    )
}
