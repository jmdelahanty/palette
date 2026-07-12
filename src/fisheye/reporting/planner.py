"""Read-only planner for composable recording and cohort reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from fisheye.shared.batch_logging import utc_now_z
from fisheye.shared.zarr_io import open_zarr_root

from .catalog import ANALYSIS_FAMILIES, PROVIDERS, VISUALIZATIONS
from .discovery import (
    RunHandle,
    choose_artifact,
    discover_chasers,
    discover_stimulus_steps,
    discover_track_ids,
    find_artifacts,
    list_family_runs,
    missing_source_requirements,
    select_family_run,
)
from .models import (
    EntityScope,
    PlanStatus,
    ProviderPlan,
    RecordingReportPlan,
    ReportPlan,
    SelectedRecording,
    VisualizationPlanItem,
)
from .selection import query_report_recordings


RootOpener = Callable[[Path], Any]


def _default_root_opener(path: Path) -> Any:
    return open_zarr_root(path, mode="r")


def _provider_plan(provider_id: str, stimulus_modes: set[str]) -> ProviderPlan:
    provider = PROVIDERS[provider_id]
    normalized_supported = {mode.upper() for mode in provider.stimulus_modes}
    matching = tuple(sorted(stimulus_modes & normalized_supported))
    if provider.always_applicable:
        return ProviderPlan(
            provider_id=provider.provider_id,
            label=provider.label,
            applicable=True,
            applicability_source="always",
        )
    if matching:
        return ProviderPlan(
            provider_id=provider.provider_id,
            label=provider.label,
            applicable=True,
            applicability_source="canonical_stimulus_steps",
            matching_stimulus_modes=matching,
        )
    return ProviderPlan(
        provider_id=provider.provider_id,
        label=provider.label,
        applicable=False,
        applicability_source="no_matching_canonical_stimulus_mode",
    )


def _entity_ids_for_visualization(
    *,
    entity_scope: EntityScope,
    track_ids: tuple[str, ...],
    chaser_ids: tuple[str, ...],
    stimulus_steps: tuple[Any, ...],
    provider_modes: tuple[str, ...],
) -> tuple[str | None, ...]:
    if entity_scope == EntityScope.RECORDING:
        return (None,)
    if entity_scope == EntityScope.TRACK:
        return tuple(track_ids) if track_ids else (None,)
    if entity_scope == EntityScope.CHASER:
        return tuple(chaser_ids) if chaser_ids else (None,)
    supported = {mode.upper() for mode in provider_modes}
    values = tuple(
        str(step.step_index)
        for step in stimulus_steps
        if not supported or step.stimulus_mode.upper() in supported
    )
    return values if values else (None,)


def _missing_prerequisites(
    *,
    family_id: str,
    handles_by_family: dict[str, tuple[RunHandle, ...]],
    entity_id: str | None,
) -> tuple[str, ...]:
    family = ANALYSIS_FAMILIES[family_id]
    missing: list[str] = []
    for prerequisite_id in family.prerequisites:
        prerequisite = ANALYSIS_FAMILIES[prerequisite_id]
        prerequisite_entity = entity_id if prerequisite.entity_id_attrs else None
        selected = select_family_run(
            handles_by_family.get(prerequisite_id, ()),
            prerequisite,
            entity_id=prerequisite_entity,
        )
        if selected is None:
            missing.append(prerequisite_id)
    return tuple(missing)


def _contract_error(
    *,
    actual_contract: str | None,
    expected_contract: str | None,
    actual_renderer: str | None,
    expected_renderer: str | None,
    actual_renderer_version: str | None,
    expected_renderer_version: str | None,
) -> str | None:
    if expected_contract is not None and actual_contract != expected_contract:
        return (
            f"visualization contract is {actual_contract!r}; expected "
            f"{expected_contract!r}"
        )
    if expected_renderer is not None and actual_renderer != expected_renderer:
        return f"renderer is {actual_renderer!r}; expected {expected_renderer!r}"
    if (
        expected_renderer_version is not None
        and actual_renderer_version != expected_renderer_version
    ):
        return (
            f"renderer version is {actual_renderer_version!r}; expected "
            f"{expected_renderer_version!r}"
        )
    return None


def _plan_visualization(
    *,
    visualization_id: str,
    entity_id: str | None,
    provider_applicable: bool,
    handles_by_family: dict[str, tuple[RunHandle, ...]],
    missing_sources_by_family: dict[str, tuple[str, ...]],
) -> VisualizationPlanItem:
    visualization = VISUALIZATIONS[visualization_id]
    family = ANALYSIS_FAMILIES[visualization.analysis_family_id]
    if not provider_applicable:
        return VisualizationPlanItem(
            visualization_id=visualization.visualization_id,
            label=visualization.label,
            provider_id=visualization.provider_id,
            required=visualization.required_by_default,
            entity_scope=visualization.entity_scope.value,
            entity_id=entity_id,
            status=PlanStatus.NOT_APPLICABLE.value,
            reason="provider does not apply to the canonical stimulus modes",
            expected_visualization_contract_id=visualization.visualization_contract_id,
            expected_renderer=visualization.renderer,
            expected_renderer_version=visualization.renderer_version,
        )

    run_entity_id = entity_id if family.entity_id_attrs else None
    run = select_family_run(
        handles_by_family.get(family.family_id, ()),
        family,
        entity_id=run_entity_id,
    )
    if run is None:
        missing_prerequisites = _missing_prerequisites(
            family_id=family.family_id,
            handles_by_family=handles_by_family,
            entity_id=entity_id,
        )
        missing_sources = missing_sources_by_family.get(family.family_id, ())
        if missing_prerequisites or missing_sources:
            status = PlanStatus.BLOCKED_MISSING_SOURCE
            parts: list[str] = []
            if missing_prerequisites:
                parts.append("prerequisite analyses: " + ", ".join(missing_prerequisites))
            if missing_sources:
                parts.append("upstream sources: " + ", ".join(missing_sources))
            reason = "missing " + "; ".join(parts)
            actions = tuple(f"analyze:{value}" for value in missing_prerequisites) + tuple(
                f"resolve_source:{value}" for value in missing_sources
            )
        else:
            status = PlanStatus.NEEDS_ANALYSIS
            reason = f"no resolved {family.family_id} run"
            actions = (f"analyze:{family.family_id}",)
        return VisualizationPlanItem(
            visualization_id=visualization.visualization_id,
            label=visualization.label,
            provider_id=visualization.provider_id,
            required=visualization.required_by_default,
            entity_scope=visualization.entity_scope.value,
            entity_id=entity_id,
            status=status.value,
            reason=reason,
            expected_visualization_contract_id=visualization.visualization_contract_id,
            expected_renderer=visualization.renderer,
            expected_renderer_version=visualization.renderer_version,
            proposed_actions=actions,
        )

    artifacts = find_artifacts(
        run,
        path_pattern=visualization.selector.path_pattern,
        entity_id=entity_id,
    )
    artifact = choose_artifact(
        artifacts,
        expected_contract_id=visualization.visualization_contract_id,
        expected_renderer=visualization.renderer,
        expected_renderer_version=visualization.renderer_version,
    )
    if artifact is None:
        return VisualizationPlanItem(
            visualization_id=visualization.visualization_id,
            label=visualization.label,
            provider_id=visualization.provider_id,
            required=visualization.required_by_default,
            entity_scope=visualization.entity_scope.value,
            entity_id=entity_id,
            status=PlanStatus.NEEDS_RENDER.value,
            reason=(
                "resolved analysis run has no artifact matching "
                f"{visualization.selector.path_pattern!r}"
            ),
            expected_visualization_contract_id=visualization.visualization_contract_id,
            expected_renderer=visualization.renderer,
            expected_renderer_version=visualization.renderer_version,
            source_run=run.reference,
            proposed_actions=(f"render:{visualization.visualization_id}",),
        )

    contract_error = _contract_error(
        actual_contract=artifact.visualization_contract_id,
        expected_contract=visualization.visualization_contract_id,
        actual_renderer=artifact.renderer,
        expected_renderer=visualization.renderer,
        actual_renderer_version=artifact.renderer_version,
        expected_renderer_version=visualization.renderer_version,
    )
    if contract_error is not None:
        return VisualizationPlanItem(
            visualization_id=visualization.visualization_id,
            label=visualization.label,
            provider_id=visualization.provider_id,
            required=visualization.required_by_default,
            entity_scope=visualization.entity_scope.value,
            entity_id=entity_id,
            status=PlanStatus.CONTRACT_MISMATCH.value,
            reason=contract_error,
            expected_visualization_contract_id=visualization.visualization_contract_id,
            expected_renderer=visualization.renderer,
            expected_renderer_version=visualization.renderer_version,
            source_run=run.reference,
            artifact=artifact,
            proposed_actions=(f"render:{visualization.visualization_id}",),
        )

    return VisualizationPlanItem(
        visualization_id=visualization.visualization_id,
        label=visualization.label,
        provider_id=visualization.provider_id,
        required=visualization.required_by_default,
        entity_scope=visualization.entity_scope.value,
        entity_id=entity_id,
        status=PlanStatus.READY.value,
        reason="contracted artifact is present",
        expected_visualization_contract_id=visualization.visualization_contract_id,
        expected_renderer=visualization.renderer,
        expected_renderer_version=visualization.renderer_version,
        source_run=run.reference,
        artifact=artifact,
    )


def plan_recording_report(
    recording: SelectedRecording,
    *,
    requested_provider_ids: Sequence[str] = (),
    include_not_applicable: bool = False,
    root_opener: RootOpener = _default_root_opener,
) -> RecordingReportPlan:
    """Build a report plan for one recording without writing to it."""

    try:
        root = root_opener(Path(recording.zarr_path))
    except Exception as exc:
        return RecordingReportPlan(
            recording=recording,
            stimulus_run=None,
            stimulus_steps=(),
            stimulus_modes=(),
            track_ids=(),
            chasers=(),
            providers=(),
            items=(),
            errors=(f"failed to open Zarr read-only: {exc}",),
        )

    handles_by_family = {
        family_id: list_family_runs(root, family)
        for family_id, family in ANALYSIS_FAMILIES.items()
    }
    missing_sources_by_family = {
        family_id: missing_source_requirements(root, family)
        for family_id, family in ANALYSIS_FAMILIES.items()
    }
    stimulus_spec = ANALYSIS_FAMILIES["stimulus.metadata"]
    stimulus_handle = select_family_run(
        handles_by_family[stimulus_spec.family_id], stimulus_spec
    )
    stimulus_steps = discover_stimulus_steps(stimulus_handle)
    chasers = discover_chasers(stimulus_handle)
    stimulus_modes = tuple(sorted({step.stimulus_mode for step in stimulus_steps}))
    stimulus_mode_set = set(stimulus_modes)

    track_spec = ANALYSIS_FAMILIES["core.track_kinematics"]
    track_handle = select_family_run(
        handles_by_family[track_spec.family_id], track_spec
    )
    track_ids = discover_track_ids(track_handle)
    chaser_ids = tuple(str(chaser.chaser_index) for chaser in chasers)

    provider_plans = tuple(
        _provider_plan(provider_id, stimulus_mode_set)
        for provider_id in PROVIDERS
    )
    provider_plan_by_id = {plan.provider_id: plan for plan in provider_plans}
    if requested_provider_ids:
        selected_provider_ids = tuple(dict.fromkeys(requested_provider_ids))
    else:
        selected_provider_ids = tuple(
            plan.provider_id for plan in provider_plans if plan.applicable
        )
        if include_not_applicable:
            selected_provider_ids = tuple(PROVIDERS)

    items: list[VisualizationPlanItem] = []
    for provider_id in selected_provider_ids:
        provider = PROVIDERS[provider_id]
        provider_plan = provider_plan_by_id[provider_id]
        if not provider_plan.applicable and not include_not_applicable and not requested_provider_ids:
            continue
        for visualization_id in provider.visualization_ids:
            visualization = VISUALIZATIONS[visualization_id]
            entity_ids = _entity_ids_for_visualization(
                entity_scope=visualization.entity_scope,
                track_ids=track_ids,
                chaser_ids=chaser_ids,
                stimulus_steps=stimulus_steps,
                provider_modes=provider.stimulus_modes,
            )
            for entity_id in entity_ids:
                items.append(
                    _plan_visualization(
                        visualization_id=visualization_id,
                        entity_id=entity_id,
                        provider_applicable=provider_plan.applicable,
                        handles_by_family=handles_by_family,
                        missing_sources_by_family=missing_sources_by_family,
                    )
                )

    return RecordingReportPlan(
        recording=recording,
        stimulus_run=(stimulus_handle.reference if stimulus_handle is not None else None),
        stimulus_steps=stimulus_steps,
        stimulus_modes=stimulus_modes,
        track_ids=track_ids,
        chasers=chasers,
        providers=provider_plans,
        items=tuple(items),
    )


def build_report_plan(
    *,
    registry_path: Path,
    protocol_name: str | None = None,
    recording_ids: Sequence[str] = (),
    recording_id_contains: str | None = None,
    path_contains: str | None = None,
    zarr_use: str = "analysis",
    status: str = "active",
    limit: int | None = None,
    all_recordings: bool = False,
    provider_ids: Sequence[str] = (),
    include_not_applicable: bool = False,
    root_opener: RootOpener = _default_root_opener,
) -> ReportPlan:
    """Query the registry and construct an immutable-in-memory read-only plan."""

    unknown = sorted(set(provider_ids) - set(PROVIDERS))
    if unknown:
        raise ValueError(f"Unknown report provider(s): {', '.join(unknown)}")
    recordings = query_report_recordings(
        registry_path,
        protocol_name=protocol_name,
        recording_ids=recording_ids,
        recording_id_contains=recording_id_contains,
        path_contains=path_contains,
        zarr_use=zarr_use,
        status=status,
        limit=limit,
        all_recordings=all_recordings,
    )
    if not recordings:
        raise ValueError("Registry query selected no recordings.")
    recording_plans = tuple(
        plan_recording_report(
            recording,
            requested_provider_ids=provider_ids,
            include_not_applicable=include_not_applicable,
            root_opener=root_opener,
        )
        for recording in recordings
    )
    counts = Counter(
        item.status
        for recording_plan in recording_plans
        for item in recording_plan.items
    )
    counts[PlanStatus.ERROR.value] += sum(
        1 for recording_plan in recording_plans if recording_plan.errors
    )
    return ReportPlan(
        schema_id="palette.dataset_report_plan.v1",
        schema_version=1,
        created_at_utc=utc_now_z(),
        tool="fisheye.reporting plan",
        registry_path=str(Path(registry_path).expanduser().resolve()),
        query={
            "protocol_name": protocol_name,
            "recording_ids": list(recording_ids),
            "recording_id_contains": recording_id_contains,
            "path_contains": path_contains,
            "zarr_use": zarr_use,
            "status": status,
            "limit": limit,
            "all_recordings": all_recordings,
            "ordering": [
                "recording_started_utc_or_recording_id",
                "arena_id",
                "recording_id",
            ],
        },
        requested_provider_ids=(
            tuple(provider_ids) if provider_ids else ("auto",)
        ),
        recordings=recording_plans,
        status_counts=dict(sorted(counts.items())),
    )
