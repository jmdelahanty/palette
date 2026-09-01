#!/usr/bin/env python3
"""Read-only, capability-routed explorer for one Palette recording Zarr.

A direct ``--zarr-path`` launch shows only that recording. Collection browsing
is enabled explicitly with ``--recordings-root`` or ``--registry``.
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from pathlib import Path
    import time

    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go

    from apps.marimo.components.analysis_catalog import (
        PROVIDERS,
        analyses_for_provider,
        group_specs_by_provider,
    )
    from apps.marimo.components.core_behavior import (
        TRACK_KINEMATICS_RENDERER,
        CoreBehaviorSource,
        ValidatedCoreBehaviorSource,
        build_core_behavior_output,
        discover_core_behavior_options,
        load_core_behavior_projection,
        validated_core_behavior_option,
    )
    from fisheye.analysis_workflows.validated_recording_behavior_source import (
        ValidatedRecordingBehaviorSource,
    )
    from apps.marimo.components.bout_kinematics import (
        available_bout_analysis_ids,
        build_bout_controls,
        build_bout_kinematics_output,
        load_bout_metric_projection,
    )
    from apps.marimo.components.goodcopbadcop_chaser import (
        available_chaser_analysis_ids,
        build_arena_heatmap,
        build_chaser_gaze_tracking_output,
        build_controls as build_chaser_controls,
        build_cra_near_field_output,
        build_cra_primary_endpoint_output,
        build_detection_occupancy_output,
        build_distance_figure,
        build_egocentric_alignment_output,
        build_egocentric_bearing_output,
        build_egocentric_polar_heatmap_output,
        build_epoch_summary_output,
        build_escape_freeze_output,
        build_fish_heading_output,
        build_spatial_occupancy_output,
        discover_chaser_gaze_tracking_components,
        load_chaser_gaze_tracking_view,
        load_goodcopbadcop_view,
        resolve_time_window_from_widgets,
        resolve_time_windows_from_multiselect,
    )
    from apps.marimo.components.provenance import build_spec_provenance_panel
    from apps.marimo.components.provider_chaser_candidate import (
        available_provider_chaser_candidate_analysis_ids,
        build_provider_chaser_candidate_bearing_output,
        build_provider_chaser_candidate_bout_response_output,
        load_provider_chaser_candidate_projection,
    )
    from apps.marimo.components.chaser_exact import (
        EXACT_CHASER_PROVIDER_ADAPTER,
    )
    from apps.marimo.components.recording_workspace import (
        RecordingExplorationWorkspace,
    )
    from apps.marimo.components.registry import (
        discover_recording_explorer_spec_options,
        discover_protocol_recording_options,
    )
    from apps.marimo.components.static_artifacts import build_static_artifacts_panel
    from apps.marimo.components.tail_kinematics import build_tail_kinematics_output

    return (
        CoreBehaviorSource,
        ValidatedCoreBehaviorSource,
        ValidatedRecordingBehaviorSource,
        EXACT_CHASER_PROVIDER_ADAPTER,
        PROVIDERS,
        Path,
        RecordingExplorationWorkspace,
        TRACK_KINEMATICS_RENDERER,
        analyses_for_provider,
        available_bout_analysis_ids,
        available_chaser_analysis_ids,
        available_provider_chaser_candidate_analysis_ids,
        build_arena_heatmap,
        build_bout_controls,
        build_bout_kinematics_output,
        build_chaser_controls,
        build_chaser_gaze_tracking_output,
        build_core_behavior_output,
        build_cra_near_field_output,
        build_cra_primary_endpoint_output,
        build_detection_occupancy_output,
        build_distance_figure,
        build_egocentric_alignment_output,
        build_egocentric_bearing_output,
        build_egocentric_polar_heatmap_output,
        build_epoch_summary_output,
        build_escape_freeze_output,
        build_fish_heading_output,
        build_provider_chaser_candidate_bearing_output,
        build_provider_chaser_candidate_bout_response_output,
        build_spatial_occupancy_output,
        build_spec_provenance_panel,
        build_static_artifacts_panel,
        build_tail_kinematics_output,
        discover_chaser_gaze_tracking_components,
        discover_core_behavior_options,
        discover_protocol_recording_options,
        discover_recording_explorer_spec_options,
        go,
        group_specs_by_provider,
        load_bout_metric_projection,
        load_chaser_gaze_tracking_view,
        load_core_behavior_projection,
        load_goodcopbadcop_view,
        load_provider_chaser_candidate_projection,
        mo,
        px,
        resolve_time_window_from_widgets,
        resolve_time_windows_from_multiselect,
        time,
        validated_core_behavior_option,
    )


@app.cell(hide_code=True)
def _(
    Path,
    ValidatedRecordingBehaviorSource,
    discover_protocol_recording_options,
    mo,
):
    cli_args = mo.cli_args()
    zarr_path_raw = cli_args.get("zarr-path")
    if not zarr_path_raw:
        raise ValueError(
            "Required --zarr-path is missing. Run: scripts/py -m marimo run "
            "apps/marimo/palette_explorer.py -- --zarr-path <analysis.zarr>"
        )
    seed_zarr_path = Path(str(zarr_path_raw))
    if not seed_zarr_path.is_dir():
        raise ValueError(f"Recording Zarr directory was not found: {seed_zarr_path}")
    recordings_root_raw = cli_args.get("recordings-root")
    registry_raw = cli_args.get("registry")
    name_contains = cli_args.get("recording-name-contains")
    initial_renderer = cli_args.get("renderer")
    initial_run_path = cli_args.get("run-path")
    initial_artifact = cli_args.get("artifact")
    exact_chaser_receipt_raw = cli_args.get("exact-chaser-receipt")
    exact_chaser_receipt_path = (
        Path(str(exact_chaser_receipt_raw)).expanduser().resolve()
        if exact_chaser_receipt_raw
        else None
    )
    if (
        exact_chaser_receipt_path is not None
        and not exact_chaser_receipt_path.is_file()
    ):
        raise ValueError(
            "Exact-chaser projection receipt was not found: "
            f"{exact_chaser_receipt_path}"
        )
    validated_behavior_bundle_raw = cli_args.get("validated-behavior-bundle")
    validated_behavior_source = (
        ValidatedRecordingBehaviorSource(
            Path(str(validated_behavior_bundle_raw)).expanduser().resolve()
        )
        if validated_behavior_bundle_raw
        else None
    )
    if validated_behavior_source is not None:
        validated_behavior_source.exact_projection_receipt_path(
            explicit_path=exact_chaser_receipt_path
        )
    workspace_mode = str(cli_args.get("workspace") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    collection_browsing = recordings_root_raw is not None or registry_raw is not None
    recording_options = discover_protocol_recording_options(
        seed_zarr_path,
        recordings_root=Path(str(recordings_root_raw)) if recordings_root_raw else None,
        registry_path=Path(str(registry_raw)) if registry_raw else None,
        renderer_filter=str(initial_renderer) if initial_renderer else None,
        run_path_filter=str(initial_run_path) if initial_run_path else None,
        artifact_filter=str(initial_artifact) if initial_artifact else None,
        name_contains=str(name_contains) if name_contains else None,
        recording_explorer_only=True,
        include_collection=collection_browsing,
        include_seed_without_specs=True,
    )
    if not recording_options:
        raise ValueError(f"No recording Zarrs were discovered from {seed_zarr_path}")
    if registry_raw:
        recording_scope_label = f"Registry collection: `{registry_raw}`"
    elif recordings_root_raw:
        recording_scope_label = f"Recording collection: `{recordings_root_raw}`"
    else:
        recording_scope_label = f"Selected recording only: `{seed_zarr_path}`"
    return (
        exact_chaser_receipt_path,
        initial_artifact,
        initial_renderer,
        initial_run_path,
        recording_options,
        recording_scope_label,
        seed_zarr_path,
        validated_behavior_source,
        workspace_mode,
    )


@app.cell(hide_code=True)
def _(mo, recording_options, recording_scope_label, seed_zarr_path):
    recording_by_label = {
        f"{index + 1}. {option.label}": option
        for index, option in enumerate(recording_options)
    }
    seed_resolved = seed_zarr_path.expanduser().resolve()
    default_recording = next(
        (
            label
            for label, option in recording_by_label.items()
            if option.zarr_path.expanduser().resolve() == seed_resolved
        ),
        next(iter(recording_by_label)),
    )
    recording_picker = mo.ui.dropdown(
        options=list(recording_by_label), value=default_recording, label="Recording"
    )
    mo.vstack(
        [
            mo.md("# Palette Recording Explorer"),
            mo.md(
                "The recording Zarr is opened read-only. A selected analysis triggers a bounded "
                "Zarr projection; exported Parquet datasets use true Polars lazy scans."
            ),
            recording_picker,
            mo.md(f"{recording_scope_label} · {len(recording_options):,} recording(s)"),
        ]
    )
    return recording_by_label, recording_picker


@app.cell(hide_code=True)
def _(
    TRACK_KINEMATICS_RENDERER,
    discover_core_behavior_options,
    discover_recording_explorer_spec_options,
    initial_artifact,
    initial_renderer,
    initial_run_path,
    recording_by_label,
    recording_picker,
    validated_behavior_source,
    validated_core_behavior_option,
):
    selected_recording = recording_by_label[recording_picker.value]
    zarr_path = selected_recording.zarr_path
    spec_options = discover_recording_explorer_spec_options(
        zarr_path,
        renderer_filter=str(initial_renderer) if initial_renderer else None,
        run_path_filter=str(initial_run_path) if initial_run_path else None,
        artifact_filter=str(initial_artifact) if initial_artifact else None,
    )
    core_options = (
        discover_core_behavior_options(zarr_path, spec_options)
        if not initial_renderer or str(initial_renderer) == TRACK_KINEMATICS_RENDERER
        else []
    )
    selected_validated_behavior_source = None
    if (
        validated_behavior_source is not None
        and validated_behavior_source.analysis_zarr == zarr_path.expanduser().resolve()
    ):
        selected_validated_behavior_source = validated_behavior_source
        core_options = [
            validated_core_behavior_option(validated_behavior_source),
            *core_options,
        ]
    return (
        core_options,
        selected_recording,
        selected_validated_behavior_source,
        spec_options,
        zarr_path,
    )


@app.cell(hide_code=True)
def _(
    PROVIDERS,
    core_options,
    group_specs_by_provider,
    mo,
    spec_options,
    zarr_path,
):
    specs_by_provider = group_specs_by_provider(spec_options)
    if core_options:
        specs_by_provider["core_behavior"] = list(core_options)
    provider_by_label = {
        PROVIDERS[provider_id].label: PROVIDERS[provider_id]
        for provider_id in PROVIDERS
        if specs_by_provider.get(provider_id)
    }
    if provider_by_label:
        default_provider = (
            PROVIDERS["stimulus_chaser"].label
            if PROVIDERS["stimulus_chaser"].label in provider_by_label
            else next(iter(provider_by_label))
        )
        provider_picker = mo.ui.dropdown(
            options=list(provider_by_label),
            value=default_provider,
            label="Analysis class",
        )
        provider_output = mo.vstack(
            [
                mo.md(f"## Available analyses\n\n`{zarr_path}`"),
                provider_picker,
                mo.md(
                    "Only analysis classes supported by persisted artifacts in this recording are shown."
                ),
            ]
        )
    else:
        provider_picker = None
        provider_output = mo.md(
            "No supported canonical analysis runs or interactive visualization specs "
            "are present in this recording."
        )
    provider_output
    return provider_by_label, provider_picker, specs_by_provider


@app.cell(hide_code=True)
def _(provider_by_label, provider_picker, specs_by_provider):
    if provider_picker is None:
        selected_provider = None
        provider_specs = []
    else:
        selected_provider = provider_by_label[provider_picker.value]
        provider_specs = specs_by_provider[selected_provider.provider_id]
    return provider_specs, selected_provider


@app.cell(hide_code=True)
def _(EXACT_CHASER_PROVIDER_ADAPTER, mo, provider_specs, selected_provider):
    if selected_provider is None or not provider_specs:
        source_by_label = {}
        source_picker = None
        source_output = mo.md("")
    else:
        source_by_label = {
            f"{index + 1}. {getattr(option, 'label', None) or option.run_name or option.run_path}": option
            for index, option in enumerate(provider_specs)
        }
        source_labels = list(source_by_label)
        requires_explicit_choice = (
            selected_provider.provider_id == EXACT_CHASER_PROVIDER_ADAPTER.provider_id
            and len(source_labels) > 1
        )
        initial_source_label = (
            EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(source_labels)
            if selected_provider.provider_id
            == EXACT_CHASER_PROVIDER_ADAPTER.provider_id
            else next(iter(source_by_label))
        )
        source_picker = mo.ui.dropdown(
            options=source_labels,
            value=initial_source_label,
            allow_select_none=requires_explicit_choice,
            label="Analysis run",
        )
        selection_guidance = (
            mo.callout(
                mo.md(
                    "Multiple immutable exact-successor bundles are available. "
                    "Choose one explicitly; no analysis arrays will load until "
                    "you do."
                ),
                kind="warn",
            )
            if requires_explicit_choice
            else mo.md("")
        )
        source_output = mo.vstack(
            [
                mo.md(
                    f"### {selected_provider.label}\n\n{selected_provider.description}"
                ),
                selection_guidance,
                source_picker,
            ]
        )
    source_output
    return source_by_label, source_picker


@app.cell(hide_code=True)
def _(source_by_label, source_picker):
    selected_spec = (
        source_by_label[source_picker.value]
        if source_picker is not None and source_picker.value is not None
        else None
    )
    return (selected_spec,)


@app.cell(hide_code=True)
def _(
    CoreBehaviorSource,
    EXACT_CHASER_PROVIDER_ADAPTER,
    ValidatedCoreBehaviorSource,
    analyses_for_provider,
    available_bout_analysis_ids,
    available_chaser_analysis_ids,
    available_provider_chaser_candidate_analysis_ids,
    mo,
    selected_provider,
    selected_spec,
    selected_validated_behavior_source,
    zarr_path,
):
    core_source = None
    if selected_provider is None or selected_spec is None:
        available_ids = ()
    elif selected_provider.provider_id == "core_behavior":
        if getattr(selected_spec, "validated_bundle_path", None) is not None:
            if selected_validated_behavior_source is None:
                raise ValueError(
                    "Selected validated Core Behavior source has no current bundle handle"
                )
            if (
                selected_spec.validated_bundle_path
                != str(selected_validated_behavior_source.bundle_path)
                or selected_spec.validated_bundle_sha256
                != selected_validated_behavior_source.bundle_sha256
            ):
                raise ValueError(
                    "Selected validated Core Behavior option belongs to an earlier "
                    "bundle identity"
                )
            core_source = ValidatedCoreBehaviorSource(
                selected_validated_behavior_source
            )
        else:
            core_source = CoreBehaviorSource(zarr_path, selected_spec)
        available_ids = core_source.available_analysis_ids()
    elif selected_provider.provider_id == "bout_kinematics":
        available_ids = available_bout_analysis_ids(zarr_path, selected_spec)
    elif selected_provider.provider_id == "stimulus_chaser_candidate":
        available_ids = available_provider_chaser_candidate_analysis_ids(
            zarr_path, selected_spec
        )
    elif selected_provider.provider_id == "stimulus_chaser_exact_successors":
        available_ids = EXACT_CHASER_PROVIDER_ADAPTER.available_analysis_ids(
            zarr_path, selected_spec
        )
    else:
        available_ids = available_chaser_analysis_ids(zarr_path, selected_spec)
    definitions = (
        {
            item.analysis_id: item
            for item in analyses_for_provider(selected_provider.provider_id)
        }
        if selected_provider is not None
        else {}
    )
    analysis_by_label = {
        definitions[analysis_id].label: definitions[analysis_id]
        for analysis_id in available_ids
        if analysis_id in definitions
    }
    if analysis_by_label:
        analysis_picker = mo.ui.dropdown(
            options=list(analysis_by_label),
            value=next(iter(analysis_by_label)),
            label="Analysis",
        )
        analysis_output = analysis_picker
    else:
        analysis_picker = None
        analysis_output = mo.md(
            "No analysis in this class has all required persisted inputs."
        )
    analysis_output
    return analysis_by_label, analysis_picker, core_source


@app.cell(hide_code=True)
def _(analysis_by_label, analysis_picker):
    selected_analysis = (
        analysis_by_label[analysis_picker.value]
        if analysis_picker is not None
        else None
    )
    selected_analysis_id = (
        selected_analysis.analysis_id if selected_analysis is not None else ""
    )
    return selected_analysis, selected_analysis_id


@app.cell(hide_code=True)
def _(
    build_bout_controls,
    mo,
    selected_analysis_id,
    selected_provider,
    selected_spec,
    zarr_path,
):
    bout_controls_error = None
    bout_controls = None
    if (
        selected_provider is not None
        and selected_provider.provider_id == "bout_kinematics"
        and selected_spec is not None
        and selected_analysis_id in {"heading", "movement", "eye_gaze"}
    ):
        try:
            bout_controls = build_bout_controls(
                mo,
                zarr_path=zarr_path,
                selected_option=selected_spec,
                analysis_id=selected_analysis_id,
            )
            bout_controls_output = (
                bout_controls.view
                if bout_controls is not None
                else mo.md("No interactive bout metrics are declared by this spec.")
            )
        except Exception as exc:
            bout_controls_error = f"{type(exc).__name__}: {exc}"
            bout_controls_output = mo.callout(
                f"Bout controls could not be created: `{bout_controls_error}`",
                kind="danger",
            )
    else:
        bout_controls_output = mo.md("")
    bout_controls_output
    return bout_controls, bout_controls_error


@app.cell(hide_code=True)
def _(
    bout_controls,
    load_bout_metric_projection,
    selected_analysis_id,
    selected_spec,
    zarr_path,
):
    bout_projection = None
    bout_projection_error = None
    if bout_controls is not None and selected_spec is not None:
        try:
            bout_metric = bout_controls.metric_by_label[
                bout_controls.metric_picker.value
            ]
            bout_heading_level = (
                str(bout_controls.heading_level_picker.value)
                if bout_controls.heading_level_picker is not None
                else None
            )
            bout_projection = load_bout_metric_projection(
                zarr_path,
                selected_spec,
                analysis_id=selected_analysis_id,
                metric=bout_metric,
                heading_level=bout_heading_level,
                bins=int(bout_controls.bins_picker.value),
                valid_only=bool(bout_controls.valid_only.value),
            )
        except Exception as exc:
            bout_projection_error = f"{type(exc).__name__}: {exc}"
    return bout_projection, bout_projection_error


@app.cell(hide_code=True)
def _(core_source, mo, selected_analysis_id):
    if core_source is not None and selected_analysis_id == "eye_angles":
        eye_run_by_label = {
            option.label: option for option in core_source.eye_angle_options()
        }
        eye_run_picker = mo.ui.dropdown(
            options=list(eye_run_by_label),
            value=next(iter(eye_run_by_label)),
            label="Eye-angle run",
        )
        eye_run_output = eye_run_picker
    else:
        eye_run_by_label = {}
        eye_run_picker = None
        eye_run_output = mo.md("")
    eye_run_output
    return eye_run_by_label, eye_run_picker


@app.cell(hide_code=True)
def _(core_source, eye_run_by_label, eye_run_picker, mo, selected_analysis_id):
    if (
        core_source is not None
        and selected_analysis_id == "eye_angles"
        and eye_run_picker is not None
    ):
        selected_eye_run = eye_run_by_label[eye_run_picker.value]
        eye_representations = list(
            core_source.eye_representations_for(selected_eye_run.run_name)
        )
        preferred = str(selected_eye_run.preferred_angle_family or "")
        default_representation = (
            preferred if preferred in eye_representations else eye_representations[0]
        )
        eye_representation_picker = mo.ui.dropdown(
            options=eye_representations,
            value=default_representation,
            label="Angle representation",
        )
        eye_representation_output = eye_representation_picker
    else:
        selected_eye_run = None
        eye_representation_picker = None
        eye_representation_output = mo.md("")
    eye_representation_output
    return eye_representation_picker, selected_eye_run


@app.cell(hide_code=True)
def _(core_source, mo, selected_analysis_id):
    if core_source is not None and selected_analysis_id == "tail_kinematics":
        tail_run_by_label = {
            option.label: option for option in core_source.tail_kinematics_options()
        }
        tail_run_picker = mo.ui.dropdown(
            options=list(tail_run_by_label),
            value=next(iter(tail_run_by_label)),
            label="Tail-kinematics run",
        )
        selected_tail_run = tail_run_by_label[tail_run_picker.value]
        tail_run_output = tail_run_picker
    else:
        tail_run_picker = None
        selected_tail_run = None
        tail_run_output = mo.md("")
    tail_run_output
    return (selected_tail_run,)


@app.cell(hide_code=True)
def _(
    core_source,
    eye_representation_picker,
    mo,
    selected_analysis_id,
    selected_eye_run,
    selected_tail_run,
):
    if core_source is not None and selected_analysis_id in {"speed", "heading"}:
        core_series_options = list(core_source.series_for(selected_analysis_id))
        core_series_picker = mo.ui.multiselect(
            options=core_series_options,
            value=list(core_source.default_series_for(selected_analysis_id)),
            label="Series",
        )
        core_series_output = core_series_picker
    elif (
        core_source is not None
        and selected_analysis_id == "eye_angles"
        and selected_eye_run is not None
        and eye_representation_picker is not None
    ):
        representation = str(eye_representation_picker.value)
        core_series_options = list(
            core_source.eye_series_for(selected_eye_run.run_name, representation)
        )
        core_series_picker = mo.ui.multiselect(
            options=core_series_options,
            value=list(
                core_source.default_eye_series_for(
                    selected_eye_run.run_name,
                    representation,
                )
            ),
            label="Eye-angle series",
        )
        core_series_output = core_series_picker
    elif (
        core_source is not None
        and selected_analysis_id == "tail_kinematics"
        and selected_tail_run is not None
    ):
        core_series_options = list(
            core_source.tail_scalar_series_for(selected_tail_run.run_name)
        )
        core_series_picker = mo.ui.multiselect(
            options=core_series_options,
            value=list(
                core_source.default_tail_scalar_series_for(selected_tail_run.run_name)
            ),
            label="Tail scalar traces",
        )
        core_series_output = core_series_picker
    else:
        core_series_picker = None
        core_series_output = mo.md("")
    core_series_output
    return (core_series_picker,)


@app.cell(hide_code=True)
def _(
    core_source,
    mo,
    selected_analysis_id,
    selected_eye_run,
    selected_provider,
    selected_tail_run,
):
    if (
        selected_provider is not None
        and selected_provider.provider_id == "core_behavior"
        and core_source is not None
        and selected_analysis_id
        in {
            "speed",
            "heading",
            "position",
            "eye_angles",
            "tail_kinematics",
            "swim_bouts",
        }
    ):
        if selected_analysis_id == "eye_angles" and selected_eye_run is not None:
            core_start_s, core_stop_s = core_source.eye_time_bounds(
                selected_eye_run.run_name
            )
            core_default_stop_s = min(core_stop_s, core_start_s + 60.0)
            core_time_step = max(
                min((core_stop_s - core_start_s) / 10000.0, 1.0),
                0.01,
            )
        elif (
            selected_analysis_id == "tail_kinematics" and selected_tail_run is not None
        ):
            core_start_s, core_stop_s = core_source.tail_time_bounds(
                selected_tail_run.run_name
            )
            core_default_stop_s = min(core_stop_s, core_start_s + 10.0)
            core_time_step = max(
                min((core_stop_s - core_start_s) / 100000.0, 0.1),
                0.001,
            )
        else:
            core_start_s, core_stop_s = core_source.time_bounds()
            core_default_stop_s = core_stop_s
            core_time_step = max((core_stop_s - core_start_s) / 1000.0, 0.001)
        core_time_window = mo.ui.range_slider(
            start=core_start_s,
            stop=max(core_stop_s, core_start_s + 1e-9),
            value=(core_start_s, core_default_stop_s),
            step=core_time_step,
            label="Time window (s)",
            full_width=True,
        )
        core_time_output = core_time_window
    else:
        core_time_window = None
        core_time_output = mo.md("")
    core_time_output
    return (core_time_window,)


@app.cell(hide_code=True)
def _(
    core_series_picker,
    core_source,
    core_time_window,
    eye_representation_picker,
    load_core_behavior_projection,
    selected_analysis_id,
    selected_eye_run,
    selected_provider,
    selected_tail_run,
):
    if (
        selected_provider is not None
        and selected_provider.provider_id == "core_behavior"
        and core_source is not None
        and selected_analysis_id
    ):
        try:
            if core_time_window is None:
                start_s = stop_s = None
            else:
                start_s, stop_s = core_time_window.value
            core_projection = load_core_behavior_projection(
                core_source,
                selected_analysis_id,
                start_s=start_s,
                stop_s=stop_s,
                series_keys=(
                    tuple(core_series_picker.value)
                    if core_series_picker is not None
                    else None
                ),
                eye_run_name=(
                    selected_eye_run.run_name if selected_eye_run is not None else None
                ),
                eye_representation=(
                    str(eye_representation_picker.value)
                    if eye_representation_picker is not None
                    else None
                ),
                tail_run_name=(
                    selected_tail_run.run_name
                    if selected_tail_run is not None
                    else None
                ),
                tail_scalar_series=(
                    tuple(core_series_picker.value)
                    if selected_analysis_id == "tail_kinematics"
                    and core_series_picker is not None
                    else None
                ),
            )
            core_error = None
        except Exception as exc:
            core_projection = None
            core_error = str(exc)
    else:
        core_projection = None
        core_error = None
    return core_error, core_projection


@app.cell(hide_code=True)
def _(
    build_core_behavior_output,
    build_tail_kinematics_output,
    core_error,
    core_projection,
    go,
    mo,
    px,
):
    if core_error:
        core_output = mo.md(f"Core behavior analysis failed: `{core_error}`")
    elif core_projection is not None:
        if core_projection.analysis_id == "tail_kinematics":
            core_output = build_tail_kinematics_output(
                mo, go, projection=core_projection
            )
        else:
            core_output = build_core_behavior_output(
                mo, go, px, projection=core_projection
            )
    else:
        core_output = mo.md("")
    core_output
    return


@app.cell(hide_code=True)
def _(
    bout_controls,
    bout_controls_error,
    bout_projection,
    bout_projection_error,
    build_bout_kinematics_output,
    go,
    mo,
    selected_analysis_id,
    selected_provider,
    selected_spec,
    zarr_path,
):
    if (
        selected_provider is not None
        and selected_provider.provider_id == "bout_kinematics"
        and selected_spec is not None
        and selected_analysis_id
    ):
        if bout_controls_error or bout_projection_error:
            bout_output = mo.callout(
                f"Bout metric projection failed: `{bout_controls_error or bout_projection_error}`",
                kind="danger",
            )
        else:
            bout_output = build_bout_kinematics_output(
                mo,
                zarr_path=zarr_path,
                selected_option=selected_spec,
                analysis_id=selected_analysis_id,
                go=go,
                projection=bout_projection,
                show_snapshot=(
                    bool(bout_controls.show_snapshot.value)
                    if bout_controls is not None
                    else False
                ),
            )
    else:
        bout_output = mo.md("")
    bout_output
    return


@app.cell(hide_code=True)
def _(
    discover_chaser_gaze_tracking_components,
    load_chaser_gaze_tracking_view,
    load_goodcopbadcop_view,
    selected_analysis_id,
    selected_provider,
    selected_spec,
    time,
    zarr_path,
):
    chaser_gaze_view = None
    chaser_needs_loaded = selected_analysis_id not in {
        "",
        "gaze_tracking",
        "static_artifacts",
        "provenance",
    }
    if (
        selected_provider is not None
        and selected_provider.provider_id == "stimulus_chaser"
        and selected_spec is not None
    ):
        try:
            if selected_analysis_id == "gaze_tracking":
                gaze_rows = discover_chaser_gaze_tracking_components(
                    zarr_path,
                    distance_run_path=selected_spec.run_path,
                )
                if not gaze_rows:
                    raise ValueError(
                        "No complete chaser-gaze component is attached to this run."
                    )
                chaser_gaze_view = load_chaser_gaze_tracking_view(
                    zarr_path,
                    str(gaze_rows[0]["component_path"]),
                )
                chaser_loaded = None
            elif chaser_needs_loaded:
                chaser_loaded = load_goodcopbadcop_view(
                    zarr_path,
                    selected_spec,
                    timer=time,
                    include_companion_analyses=selected_analysis_id
                    in {"cra_quadrant", "cra_near_field", "escape_freeze"},
                    analysis_id=selected_analysis_id,
                )
            else:
                chaser_loaded = None
            chaser_error = None
        except Exception as exc:
            chaser_loaded = None
            chaser_gaze_view = None
            chaser_error = str(exc)
    else:
        chaser_loaded = None
        chaser_error = None
    return chaser_error, chaser_gaze_view, chaser_loaded


@app.cell(hide_code=True)
def _(
    load_provider_chaser_candidate_projection,
    selected_analysis_id,
    selected_provider,
    selected_spec,
    zarr_path,
):
    candidate_projection = None
    candidate_projection_error = None
    if (
        selected_provider is not None
        and selected_provider.provider_id == "stimulus_chaser_candidate"
        and selected_spec is not None
        and selected_analysis_id in {"egocentric_bearing", "bout_response"}
    ):
        try:
            candidate_projection = load_provider_chaser_candidate_projection(
                zarr_path,
                selected_spec,
                require_bout=selected_analysis_id == "bout_response",
            )
        except Exception as exc:
            candidate_projection_error = f"{type(exc).__name__}: {exc}"
    return candidate_projection, candidate_projection_error


@app.cell(hide_code=True)
def _(
    EXACT_CHASER_PROVIDER_ADAPTER,
    exact_chaser_receipt_path,
    selected_analysis_id,
    selected_provider,
    selected_spec,
    selected_validated_behavior_source,
    zarr_path,
):
    exact_chaser_projection = None
    exact_chaser_projection_error = None
    if (
        selected_provider is not None
        and selected_provider.provider_id == EXACT_CHASER_PROVIDER_ADAPTER.provider_id
        and selected_spec is not None
        and bool(selected_analysis_id)
        and EXACT_CHASER_PROVIDER_ADAPTER.requires_projection(selected_analysis_id)
    ):
        try:
            exact_chaser_projection = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
                zarr_path,
                selected_spec,
                analysis_id=selected_analysis_id,
                projection_receipt_path=exact_chaser_receipt_path,
                validated_behavior_source=selected_validated_behavior_source,
            )
        except Exception as exc:
            exact_chaser_projection_error = f"{type(exc).__name__}: {exc}"
    return exact_chaser_projection, exact_chaser_projection_error


@app.cell(hide_code=True)
def _(build_chaser_controls, chaser_loaded, mo, selected_analysis_id):
    analyses_with_controls = {
        "distance",
        "egocentric_bearing",
        "polar_distance",
        "fish_heading",
        "alignment",
        "position_heatmap",
        "detection_occupancy",
        "spatial_occupancy",
    }
    if chaser_loaded is not None and selected_analysis_id in analyses_with_controls:
        chaser_controls = build_chaser_controls(mo, loaded=chaser_loaded)
    else:
        chaser_controls = None
    return (chaser_controls,)


@app.cell(hide_code=True)
def _(chaser_controls, mo, selected_analysis_id):
    if chaser_controls is None:
        chaser_controls_output = mo.md("")
    else:
        items = []
        if selected_analysis_id == "distance":
            items.append(chaser_controls.distance_series_picker)
        if selected_analysis_id in {
            "distance",
            "egocentric_bearing",
            "polar_distance",
            "alignment",
        }:
            items.append(chaser_controls.chaser_picker)
        if selected_analysis_id == "egocentric_bearing":
            items.append(chaser_controls.egocentric_epoch_picker)
        else:
            items.append(chaser_controls.epoch_picker)
            if (
                chaser_controls.epoch_options.get(chaser_controls.epoch_picker.value)
                is None
            ):
                items.append(chaser_controls.time_window)
        if selected_analysis_id in {"polar_distance", "position_heatmap"}:
            items.append(chaser_controls.heatmap_bins)
        if selected_analysis_id == "position_heatmap":
            items.append(chaser_controls.chaser_overlay)
        if (
            selected_analysis_id == "spatial_occupancy"
            and chaser_controls.spatial_zone_set_picker is not None
        ):
            items.append(chaser_controls.spatial_zone_set_picker)
        chaser_controls_output = mo.vstack(items)
    chaser_controls_output
    return


@app.cell(hide_code=True)
def _(chaser_controls, chaser_loaded, resolve_time_window_from_widgets):
    if chaser_loaded is not None and chaser_controls is not None:
        chaser_window = resolve_time_window_from_widgets(
            epoch_options=chaser_controls.epoch_options,
            epoch_picker=chaser_controls.epoch_picker,
            time_window=chaser_controls.time_window,
            windows_df=chaser_loaded.windows_df,
        )
    else:
        chaser_window = None
    return (chaser_window,)


@app.cell(hide_code=True)
def _(
    chaser_controls,
    chaser_loaded,
    resolve_time_windows_from_multiselect,
    selected_analysis_id,
):
    if (
        selected_analysis_id == "egocentric_bearing"
        and chaser_loaded is not None
        and chaser_controls is not None
    ):
        chaser_egocentric_windows = resolve_time_windows_from_multiselect(
            epoch_options=chaser_controls.epoch_options,
            epoch_picker=chaser_controls.egocentric_epoch_picker,
            windows_df=chaser_loaded.windows_df,
        )
    else:
        chaser_egocentric_windows = ()
    return (chaser_egocentric_windows,)


@app.cell(hide_code=True)
def _(
    EXACT_CHASER_PROVIDER_ADAPTER,
    build_arena_heatmap,
    build_chaser_gaze_tracking_output,
    build_cra_near_field_output,
    build_cra_primary_endpoint_output,
    build_detection_occupancy_output,
    build_distance_figure,
    build_egocentric_alignment_output,
    build_egocentric_bearing_output,
    build_egocentric_polar_heatmap_output,
    build_epoch_summary_output,
    build_escape_freeze_output,
    build_fish_heading_output,
    build_provider_chaser_candidate_bearing_output,
    build_provider_chaser_candidate_bout_response_output,
    build_spatial_occupancy_output,
    build_spec_provenance_panel,
    build_static_artifacts_panel,
    candidate_projection,
    candidate_projection_error,
    chaser_controls,
    chaser_egocentric_windows,
    chaser_error,
    chaser_gaze_view,
    chaser_loaded,
    chaser_window,
    exact_chaser_projection,
    exact_chaser_projection_error,
    exact_chaser_receipt_path,
    go,
    mo,
    px,
    selected_analysis_id,
    selected_provider,
    selected_spec,
    selected_validated_behavior_source,
    zarr_path,
):
    if selected_provider is None or selected_provider.provider_id not in {
        "stimulus_chaser",
        "stimulus_chaser_candidate",
        "stimulus_chaser_exact_successors",
    }:
        chaser_output = mo.md("")
    elif chaser_error:
        chaser_output = mo.md(f"Chaser analysis failed: `{chaser_error}`")
    elif candidate_projection_error:
        chaser_output = mo.callout(
            f"Candidate projection failed: `{candidate_projection_error}`",
            kind="danger",
        )
    elif exact_chaser_projection_error:
        chaser_output = mo.callout(
            f"Exact successor projection failed closed: `{exact_chaser_projection_error}`",
            kind="danger",
        )
    elif selected_analysis_id == "static_artifacts" and selected_spec is not None:
        chaser_output = build_static_artifacts_panel(
            mo,
            zarr_path=zarr_path,
            run_path=selected_spec.run_path,
            spec=selected_spec.spec,
        )
    elif selected_analysis_id == "provenance" and selected_spec is not None:
        chaser_output = build_spec_provenance_panel(
            mo,
            spec=selected_spec.spec,
            artifact_attrs=selected_spec.attrs,
            option=selected_spec,
        )
    elif (
        selected_analysis_id == "egocentric_bearing"
        and candidate_projection is not None
    ):
        chaser_output = build_provider_chaser_candidate_bearing_output(
            mo,
            candidate_projection,
        )
    elif selected_analysis_id == "bout_response" and candidate_projection is not None:
        chaser_output = build_provider_chaser_candidate_bout_response_output(
            mo,
            candidate_projection,
        )
    elif exact_chaser_projection is not None and selected_spec is not None:
        chaser_output = EXACT_CHASER_PROVIDER_ADAPTER.render(
            mo,
            go,
            exact_chaser_projection,
            zarr_path=zarr_path,
            option=selected_spec,
            analysis_id=selected_analysis_id,
            projection_receipt_path=exact_chaser_receipt_path,
            validated_behavior_source=selected_validated_behavior_source,
        )
    elif selected_analysis_id == "gaze_tracking" and chaser_gaze_view is not None:
        chaser_output = build_chaser_gaze_tracking_output(
            mo,
            loaded=chaser_gaze_view,
        )
    elif chaser_loaded is None:
        chaser_output = mo.md("")
    elif selected_analysis_id == "distance" and chaser_window is not None:
        chaser_output, _ = build_distance_figure(
            go,
            loaded=chaser_loaded,
            distance_series_picker=chaser_controls.distance_series_picker,
            window=chaser_window,
        )
    elif selected_analysis_id == "epoch_summary":
        chaser_output = build_epoch_summary_output(
            mo, go, loaded=chaser_loaded, chaser_picker=None
        )
    elif selected_analysis_id == "egocentric_bearing":
        chaser_output = build_egocentric_bearing_output(
            mo,
            go,
            loaded=chaser_loaded,
            windows=chaser_egocentric_windows,
            chaser_picker=chaser_controls.chaser_picker,
        )
    elif selected_analysis_id == "polar_distance" and chaser_window is not None:
        chaser_output = build_egocentric_polar_heatmap_output(
            mo,
            go,
            loaded=chaser_loaded,
            window=chaser_window,
            chaser_picker=chaser_controls.chaser_picker,
        )
    elif selected_analysis_id == "fish_heading" and chaser_window is not None:
        chaser_output = build_fish_heading_output(
            mo, go, loaded=chaser_loaded, window=chaser_window
        )
    elif selected_analysis_id == "alignment" and chaser_window is not None:
        chaser_output = build_egocentric_alignment_output(
            mo,
            go,
            loaded=chaser_loaded,
            window=chaser_window,
            chaser_picker=chaser_controls.chaser_picker,
        )
    elif selected_analysis_id == "position_heatmap" and chaser_window is not None:
        chaser_output, _ = build_arena_heatmap(
            px,
            loaded=chaser_loaded,
            heatmap_bins=chaser_controls.heatmap_bins,
            chaser_overlay=chaser_controls.chaser_overlay,
            window=chaser_window,
        )
    elif selected_analysis_id == "detection_occupancy" and chaser_window is not None:
        chaser_output = build_detection_occupancy_output(
            mo, go, loaded=chaser_loaded, window=chaser_window
        )
    elif selected_analysis_id == "spatial_occupancy" and chaser_window is not None:
        chaser_output = build_spatial_occupancy_output(
            mo,
            go,
            loaded=chaser_loaded,
            window=chaser_window,
            spatial_zone_set_picker=chaser_controls.spatial_zone_set_picker,
        )
    elif selected_analysis_id == "cra_quadrant":
        chaser_output = build_cra_primary_endpoint_output(mo, go, loaded=chaser_loaded)
    elif selected_analysis_id == "cra_near_field":
        chaser_output = build_cra_near_field_output(mo, go, loaded=chaser_loaded)
    elif selected_analysis_id == "escape_freeze":
        chaser_output = build_escape_freeze_output(mo, loaded=chaser_loaded)
    else:
        chaser_output = mo.md("The selected analysis has no renderable persisted rows.")
    chaser_output
    return


@app.cell(hide_code=True)
def _(
    RecordingExplorationWorkspace,
    chaser_gaze_view,
    chaser_loaded,
    core_projection,
    core_source,
    exact_chaser_projection,
    selected_analysis,
    selected_provider,
    selected_recording,
    selected_spec,
    zarr_path,
):
    recording_workspace = RecordingExplorationWorkspace(
        zarr_path=zarr_path,
        selected_recording=selected_recording,
        selected_provider=selected_provider,
        selected_spec=selected_spec,
        selected_analysis=selected_analysis,
        core_source=core_source,
        core_projection=core_projection,
        chaser_view=(
            chaser_gaze_view
            if chaser_gaze_view is not None
            else (
                exact_chaser_projection
                if exact_chaser_projection is not None
                else chaser_loaded
            )
        ),
    )
    return (recording_workspace,)


@app.cell(hide_code=True)
def _(mo, workspace_mode):
    if workspace_mode:
        workspace_header = mo.vstack(
            [
                mo.md("---\n\n## Exploration workspace"),
                mo.callout(
                    mo.md(
                        "The selected recording is mounted read-only at "
                        "`/data/recording.zarr`. Notebook edits and new files "
                        "may be saved only beneath `/workspace`."
                    ),
                    kind="info",
                ),
                mo.md(
                    "Use the editable cell below, add more cells after it, or "
                    "ask a Marimo Pair agent to inspect `exploration` and append "
                    "an analysis. Changing the controls above updates this live handle."
                ),
            ]
        )
    else:
        workspace_header = mo.md("")
    workspace_header
    return


@app.cell
def _(recording_workspace, workspace_mode):
    # Start here. `exploration` follows the recording and analysis selected above.
    # Try `exploration.summary()`, `exploration.core_frame`,
    # `exploration.chaser_tables`, `exploration.persisted_pngs`,
    # or `exploration.open_zarr()`.
    exploration = recording_workspace if workspace_mode else None
    exploration
    return


if __name__ == "__main__":
    app.run()
