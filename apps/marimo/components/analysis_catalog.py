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
        AnalysisDefinition(
            "speed", "Speed traces", "Projected speed and acceleration series."
        ),
        AnalysisDefinition(
            "distance_traveled",
            "Distance traveled",
            "Observed cumulative path and per-second distance with tracking-gap evidence.",
        ),
        AnalysisDefinition(
            "heading", "Heading and turning", "Heading and angular-motion traces."
        ),
        AnalysisDefinition(
            "position",
            "Position and trajectory",
            "Projected x/y trajectory and occupancy.",
        ),
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
        AnalysisDefinition(
            "swim_bouts", "Swim bouts", "Persisted bout events and distributions."
        ),
        AnalysisDefinition(
            "baseline",
            "Pre-period behavior",
            "Descriptive activity and trajectory during the persisted baseline epoch.",
        ),
    ),
)


RECORDING_BEHAVIOR_DISTRIBUTION_PROVIDER = ProviderDefinition(
    provider_id="recording_behavior_distributions",
    label="Recording behavior distributions",
    description=(
        "Receipt-bound whole-session, protocol-epoch, and named-time "
        "distributions from one immutable recording run."
    ),
    component_key="recording_behavior_distributions",
    analyses=(
        AnalysisDefinition(
            "distributions",
            "Persisted behavior distributions",
            (
                "Exact persisted histogram bins, denominators, exclusions, and "
                "scope memberships without viewer-side rebinning."
            ),
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
            "Arena occupancy from the selected position authority.",
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
        AnalysisDefinition(
            "static_artifacts", "Persisted plots", "Analysis-owned PNG artifacts."
        ),
        AnalysisDefinition(
            "provenance",
            "Provenance and source rows",
            "Spec lineage and projected source rows.",
        ),
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
            "egocentric_bearing",
            "Candidate egocentric bearing",
            (
                "Descriptive chaser-bearing views using the exact unpromoted "
                "position provider and its persisted body-heading authority."
            ),
        ),
        AnalysisDefinition(
            "bout_response",
            "Candidate bout responses",
            (
                "Bout-start distance and bearing summaries joined to the exact "
                "selector-ineligible swim-bout run."
            ),
        ),
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


CHASER_EXACT_SUCCESSOR_PROVIDER = ProviderDefinition(
    provider_id="stimulus_chaser_exact_successors",
    label="Chaser exact successors",
    description=(
        "Paired keypoint/detection views from sealed, selector-ineligible "
        "protocol-semantic successors."
    ),
    component_key="chaser_exact_successors",
    analyses=(
        AnalysisDefinition(
            "radial_near_field",
            "Distance, rings, and near field",
            "Paired-provider persisted radial and exact-time near-field summaries.",
        ),
        AnalysisDefinition(
            "distance_distributions",
            "Distance distributions and controls",
            (
                "Persisted empirical CDFs, observed versus moving-reference "
                "geometric radial mass, and configured wall-excluded twins."
            ),
        ),
        AnalysisDefinition(
            "same_quadrant_occupancy",
            "Same-quadrant occupancy",
            (
                "Persisted scalar fish/chaser same-quadrant fractions with "
                "valid-row and all-candidate denominators shown separately."
            ),
        ),
        AnalysisDefinition(
            "distance_traces",
            "Full and exact-epoch distance",
            "Full-session and exact protocol-epoch fish–chaser distance traces.",
        ),
        AnalysisDefinition(
            "body_bearing_polar",
            "Body-frame bearing polar",
            (
                "Whole-circle chaser-bearing distributions from the accepted "
                "keypoint body-axis supplier, with no detection or heading fallback."
            ),
        ),
        AnalysisDefinition(
            "body_bearing_distance",
            "Body-frame bearing × distance",
            (
                "Exact anatomical-bearing point clouds and joint polar densities "
                "using receipt-bound keypoint distance and body-axis evidence."
            ),
        ),
        AnalysisDefinition(
            "fish_heading",
            "Anatomical fish heading",
            (
                "Whole-circle fish-heading distributions from the accepted "
                "keypoint body-axis supplier, collapsed once per acquisition frame."
            ),
        ),
        AnalysisDefinition(
            "trajectory_overlays",
            "Exact-epoch position overlays",
            "Fish positions with exact logged chaser positions in the reviewed arena.",
        ),
        AnalysisDefinition(
            "spatial_occupancy",
            "Exact-epoch occupancy heatmaps",
            (
                "Persisted paired-provider occupancy and detection-minus-keypoint "
                "display on the sealed reviewed-arena physical grid."
            ),
        ),
        AnalysisDefinition(
            "controller_trials",
            "Exact controller trials",
            (
                "Full-session and trigger-aligned distance from producer-logged "
                "active trial membership, with retained nonmember gaps."
            ),
        ),
        AnalysisDefinition(
            "generalized_bout_response",
            "Generalized bout response",
            (
                "Persisted bout rate, kinematics, separation response, and "
                "optional body-frame turning by exact onset distance."
            ),
        ),
        AnalysisDefinition(
            "escape_freeze",
            "Exact escape/freeze outcomes",
            (
                "Persisted speed-defined escape events, separately annotated "
                "high turns, exact-trial freeze candidates, recapture outcomes, "
                "validity reasons, and threshold sensitivity."
            ),
        ),
        AnalysisDefinition(
            "gaze_tracking",
            "Exact body-frame gaze tracking",
            (
                "Persisted eye-versus-chaser bearing, gaze error, lock fractions, "
                "static tracking gain, and sustained lock events from exact sources."
            ),
        ),
        AnalysisDefinition(
            "epoch_behavior",
            "Protocol-semantic epoch behavior",
            (
                "Persisted physical speed, path, tracking coverage, swim-bout "
                "summaries, bout-kinematics distributions, and inter-bout "
                "interval distributions for exact pre/training/post epochs."
            ),
        ),
        AnalysisDefinition(
            "body_alignment_by_distance",
            "Fish alignment by chaser distance",
            (
                "Persisted anatomical alignment, absolute and circular bearing, "
                "and explicit validity/support by exact semantic epoch, chaser, "
                "and physical-distance bin."
            ),
        ),
        AnalysisDefinition(
            "provenance",
            "Provenance",
            "Exact bundle, child-run identities, display projection, and authorities.",
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
    RECORDING_BEHAVIOR_DISTRIBUTION_PROVIDER.provider_id: (
        RECORDING_BEHAVIOR_DISTRIBUTION_PROVIDER
    ),
    CHASER_PROVIDER.provider_id: CHASER_PROVIDER,
    CHASER_CANDIDATE_PROVIDER.provider_id: CHASER_CANDIDATE_PROVIDER,
    CHASER_EXACT_SUCCESSOR_PROVIDER.provider_id: CHASER_EXACT_SUCCESSOR_PROVIDER,
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
            (
                0
                if "chaser-dashboard" in item.renderer
                or "chaser-protocol-dashboard" in item.renderer
                else 1
            ),
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
    if "stimulus_chaser_exact_successors" in grouped:
        grouped["stimulus_chaser_exact_successors"].sort(
            key=lambda item: (
                str(item.attrs.get("palette_run_completed_at_utc") or ""),
                item.run_name,
            ),
            reverse=True,
        )
    return grouped


def analyses_for_provider(provider_id: str) -> tuple[AnalysisDefinition, ...]:
    provider = PROVIDERS.get(str(provider_id))
    return provider.analyses if provider is not None else ()
