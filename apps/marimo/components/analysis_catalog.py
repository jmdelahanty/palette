"""Capability-based provider and analysis routing for the recording explorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from fisheye.visualization.bout_kinematics_interactive import BOUT_HEADING_PLOT_RENDERER

from .registry import InteractiveSpecOption, renderer_registration_for


@dataclass(frozen=True)
class AnalysisDefinition:
    analysis_id: str
    label: str
    description: str


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: str
    label: str
    description: str
    component_key: str
    analyses: tuple[AnalysisDefinition, ...]


CORE_BEHAVIOR_PROVIDER = ProviderDefinition(
    provider_id="core_behavior",
    label="Core behavior",
    description="Stimulus-independent movement and baseline views.",
    component_key="core_behavior",
    analyses=(
        AnalysisDefinition("speed", "Speed traces", "Projected speed and acceleration series."),
        AnalysisDefinition("heading", "Heading and turning", "Heading and angular-motion traces."),
        AnalysisDefinition("position", "Position and trajectory", "Projected x/y trajectory and occupancy."),
        AnalysisDefinition(
            "eye_angles",
            "Eye angles and convergence",
            "Framewise eye-angle, gaze, and convergence traces when persisted.",
        ),
        AnalysisDefinition(
            "tail_kinematics",
            "Tail posture and curvature",
            "Framewise body-frame tail angles, spline curvature, and synchronized bouts.",
        ),
        AnalysisDefinition("swim_bouts", "Swim bouts", "Persisted bout events and distributions."),
        AnalysisDefinition(
            "baseline",
            "Pre-period behavior",
            "Descriptive activity and trajectory during the persisted baseline epoch.",
        ),
    ),
)


CHASER_PROVIDER = ProviderDefinition(
    provider_id="stimulus_chaser",
    label="Chaser stimulus",
    description="Distance, spatial, egocentric, quadrant, near-field, and escape analyses.",
    component_key="goodcopbadcop_chaser",
    analyses=(
        AnalysisDefinition(
            "distance",
            "Chaser distances",
            "Per-chaser distance traces and epoch selection.",
        ),
        AnalysisDefinition(
            "epoch_summary", "Epoch behavior", "Per-epoch chaser and fish summaries."
        ),
        AnalysisDefinition(
            "egocentric_bearing",
            "Egocentric bearing",
            "Bearing distributions for selected chasers.",
        ),
        AnalysisDefinition(
            "polar_distance",
            "Polar bearing and distance",
            "Re-binnable polar small multiples.",
        ),
        AnalysisDefinition(
            "fish_heading", "Fish heading", "Heading traces over selected epochs."
        ),
        AnalysisDefinition(
            "alignment",
            "Distance and alignment",
            "Egocentric distance/alignment summaries.",
        ),
        AnalysisDefinition(
            "position_heatmap",
            "Position heatmap",
            "Arena occupancy with optional chaser overlay.",
        ),
        AnalysisDefinition(
            "detection_occupancy",
            "Detection occupancy",
            "Persisted detection occupancy heatmap.",
        ),
        AnalysisDefinition(
            "spatial_occupancy",
            "Spatial occupancy",
            "Quadrant and configured-zone occupancy.",
        ),
        AnalysisDefinition(
            "cra_quadrant",
            "Chaser quadrant occupancy",
            "Occupancy in each chaser's quadrant versus the other quadrants.",
        ),
        AnalysisDefinition(
            "cra_near_field",
            "Chaser near-field occupancy",
            "Near-field dwell, visits, distance distributions, and radial density.",
        ),
        AnalysisDefinition(
            "escape_freeze",
            "Escape outcomes",
            "Successful escape and freezing diagnostics.",
        ),
        AnalysisDefinition(
            "gaze_tracking",
            "Eye–chaser tracking",
            "Body-frame gaze tracking, rotated controls, and sustained lock-on events.",
        ),
        AnalysisDefinition("static_artifacts", "Persisted plots", "Analysis-owned PNG artifacts."),
        AnalysisDefinition("provenance", "Provenance and source rows", "Spec lineage and projected source rows."),
    ),
)


CHASER_CANDIDATE_PROVIDER = ProviderDefinition(
    provider_id="stimulus_chaser_candidate",
    label="Chaser provider candidates (unpromoted)",
    description=(
        "Read-only comparison candidates. These runs are manifest-validated but "
        "remain selector-ineligible until a scientific promotion decision."
    ),
    component_key="provider_chaser_candidate",
    analyses=(
        AnalysisDefinition(
            "static_artifacts",
            "Candidate plots",
            "Persisted stimulus-sample distance traces and histograms.",
        ),
        AnalysisDefinition(
            "provenance",
            "Candidate provenance",
            "Exact manifest, provider lineage, coordinate authority, and candidate status.",
        ),
    ),
)


BOUT_KINEMATICS_PROVIDER = ProviderDefinition(
    provider_id="bout_kinematics",
    label="Bout kinematics",
    description="Persisted per-bout heading, movement, and eye-gaze summaries.",
    component_key="bout_kinematics",
    analyses=(
        AnalysisDefinition(
            "heading",
            "Heading kinematics",
            "Net heading change and within-bout angular-motion summaries.",
        ),
        AnalysisDefinition(
            "movement",
            "Physical movement",
            "Bout duration, physical path length, and speed summaries.",
        ),
        AnalysisDefinition(
            "eye_gaze",
            "Eye gaze and convergence",
            "Bout-aligned eye-gaze, convergence, and validity summaries when persisted.",
        ),
        AnalysisDefinition(
            "provenance",
            "Provenance",
            "Visualization specs, parameters, source paths, and artifact metadata.",
        ),
    ),
)


PROVIDERS: Mapping[str, ProviderDefinition] = {
    CORE_BEHAVIOR_PROVIDER.provider_id: CORE_BEHAVIOR_PROVIDER,
    CHASER_PROVIDER.provider_id: CHASER_PROVIDER,
    CHASER_CANDIDATE_PROVIDER.provider_id: CHASER_CANDIDATE_PROVIDER,
    BOUT_KINEMATICS_PROVIDER.provider_id: BOUT_KINEMATICS_PROVIDER,
}


def provider_id_for_component(component_key: str) -> str | None:
    for provider in PROVIDERS.values():
        if provider.component_key == component_key:
            return provider.provider_id
    return None


def group_specs_by_provider(
    options: Iterable[InteractiveSpecOption],
) -> dict[str, list[InteractiveSpecOption]]:
    """Group supported specs by capability provider, never by protocol name."""

    grouped: dict[str, list[InteractiveSpecOption]] = {}
    seen_paths: dict[str, set[str]] = {}
    ordered = sorted(
        options,
        key=lambda item: (
            0 if "chaser-dashboard" in item.renderer or "chaser-protocol-dashboard" in item.renderer else 1,
            0 if item.renderer == BOUT_HEADING_PLOT_RENDERER else 1,
            item.artifact_path,
        ),
    )
    for option in ordered:
        registration = renderer_registration_for(option.renderer)
        if registration is None:
            continue
        provider_id = provider_id_for_component(registration.component_key)
        if provider_id is None:
            continue
        # Companion chaser specs route to the owning chaser-distance run. Keep a
        # single selectable source for that run instead of repeating every spec.
        run_key = option.run_path
        if "/components/" in run_key:
            run_key = run_key.split("/components/", 1)[0]
        provider_seen = seen_paths.setdefault(provider_id, set())
        if run_key in provider_seen:
            continue
        provider_seen.add(run_key)
        grouped.setdefault(provider_id, []).append(option)
    if "bout_kinematics" in grouped:
        grouped["bout_kinematics"].sort(
            key=lambda item: (
                str(item.attrs.get("created_at_utc") or ""),
                item.run_name,
            ),
            reverse=True,
        )
    if "stimulus_chaser_candidate" in grouped:
        grouped["stimulus_chaser_candidate"].sort(
            key=lambda item: (
                str(item.attrs.get("created_at_utc") or ""),
                item.run_name,
            ),
            reverse=True,
        )
    return grouped


def analyses_for_provider(provider_id: str) -> tuple[AnalysisDefinition, ...]:
    provider = PROVIDERS.get(str(provider_id))
    return provider.analyses if provider is not None else ()
