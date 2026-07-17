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
        build_core_behavior_output,
        discover_core_behavior_options,
        load_core_behavior_projection,
    )
    from apps.marimo.components.goodcopbadcop_chaser import (
        available_chaser_analysis_ids,
        build_arena_heatmap,
        build_chaser_gaze_tracking_output,
        build_controls as build_chaser_controls,
        build_cra_near_field_output,
        build_cra_primary_endpoint_output,
        build_debug_tables,
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
    from apps.marimo.components.recording_workspace import (
        RecordingExplorationWorkspace,
    )
    from apps.marimo.components.registry import (
        discover_recording_explorer_spec_options,
        discover_protocol_recording_options,
    )
    from apps.marimo.components.static_artifacts import build_static_artifacts_panel

    return (
        PROVIDERS,
        TRACK_KINEMATICS_RENDERER,
        CoreBehaviorSource,
        Path,
        RecordingExplorationWorkspace,
        analyses_for_provider,
        available_chaser_analysis_ids,
        build_arena_heatmap,
        build_chaser_gaze_tracking_output,
        build_chaser_controls,
        build_core_behavior_output,
        build_cra_near_field_output,
        build_cra_primary_endpoint_output,
        build_debug_tables,
        build_detection_occupancy_output,
        build_distance_figure,
        build_egocentric_alignment_output,
        build_egocentric_bearing_output,
        build_egocentric_polar_heatmap_output,
        build_epoch_summary_output,
        build_escape_freeze_output,
        build_fish_heading_output,
        build_spatial_occupancy_output,
        build_spec_provenance_panel,
        build_static_artifacts_panel,
        discover_recording_explorer_spec_options,
        discover_core_behavior_options,
        discover_protocol_recording_options,
        discover_chaser_gaze_tracking_components,
        go,
        group_specs_by_provider,
        load_core_behavior_projection,
        load_chaser_gaze_tracking_view,
        load_goodcopbadcop_view,
        mo,
        px,
        resolve_time_window_from_widgets,
        resolve_time_windows_from_multiselect,
        time,
    )


@app.cell(hide_code=True)
def _(Path, discover_protocol_recording_options, mo):
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
        initial_artifact,
        initial_renderer,
        initial_run_path,
        recording_scope_label,
        recording_options,
        seed_zarr_path,
        workspace_mode,
    )


@app.cell(hide_code=True)
def _(mo, recording_options, recording_scope_label, seed_zarr_path):
    recording_by_label = {
        f"{index + 1}. {option.label}": option for index, option in enumerate(recording_options)
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
            mo.md(
                f"{recording_scope_label} · {len(recording_options):,} recording(s)"
            ),
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
    return core_options, selected_recording, spec_options, zarr_path


@app.cell(hide_code=True)
def _(PROVIDERS, core_options, group_specs_by_provider, mo, spec_options, zarr_path):
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
            options=list(provider_by_label), value=default_provider, label="Analysis class"
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
def _(mo, provider_specs, selected_provider):
    if selected_provider is None or not provider_specs:
        source_by_label = {}
        source_picker = None
        source_output = mo.md("")
    else:
        source_by_label = {
            f"{index + 1}. {getattr(option, 'label', None) or option.run_name or option.run_path}": option
            for index, option in enumerate(provider_specs)
        }
        source_picker = mo.ui.dropdown(
            options=list(source_by_label), value=next(iter(source_by_label)), label="Analysis run"
        )
        source_output = mo.vstack(
            [
                mo.md(f"### {selected_provider.label}\n\n{selected_provider.description}"),
                source_picker,
            ]
        )
    source_output
    return source_by_label, source_picker


@app.cell(hide_code=True)
def _(source_by_label, source_picker):
    selected_spec = source_by_label[source_picker.value] if source_picker is not None else None
    return (selected_spec,)


@app.cell(hide_code=True)
def _(
    CoreBehaviorSource,
    analyses_for_provider,
    available_chaser_analysis_ids,
    mo,
    selected_provider,
    selected_spec,
    zarr_path,
):
    core_source = None
    if selected_provider is None or selected_spec is None:
        available_ids = ()
    elif selected_provider.provider_id == "core_behavior":
        core_source = CoreBehaviorSource(zarr_path, selected_spec)
        available_ids = core_source.available_analysis_ids()
    else:
        available_ids = available_chaser_analysis_ids(zarr_path, selected_spec)
    definitions = {
        item.analysis_id: item for item in analyses_for_provider(selected_provider.provider_id)
    } if selected_provider is not None else {}
    analysis_by_label = {
        definitions[analysis_id].label: definitions[analysis_id]
        for analysis_id in available_ids
        if analysis_id in definitions
    }
    if analysis_by_label:
        analysis_picker = mo.ui.dropdown(
            options=list(analysis_by_label), value=next(iter(analysis_by_label)), label="Analysis"
        )
        analysis_output = analysis_picker
    else:
        analysis_picker = None
        analysis_output = mo.md("No analysis in this class has all required persisted inputs.")
    analysis_output
    return analysis_by_label, analysis_picker, core_source


@app.cell(hide_code=True)
def _(analysis_by_label, analysis_picker):
    selected_analysis = (
        analysis_by_label[analysis_picker.value] if analysis_picker is not None else None
    )
    selected_analysis_id = selected_analysis.analysis_id if selected_analysis is not None else ""
    return selected_analysis, selected_analysis_id


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
def _(
    core_source,
    eye_representation_picker,
    mo,
    selected_analysis_id,
    selected_eye_run,
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
    else:
        core_series_picker = None
        core_series_output = mo.md("")
    core_series_output
    return (core_series_picker,)


@app.cell(hide_code=True)
def _(core_source, mo, selected_analysis_id, selected_eye_run, selected_provider):
    if (
        selected_provider is not None
        and selected_provider.provider_id == "core_behavior"
        and core_source is not None
        and selected_analysis_id in {"speed", "heading", "position", "eye_angles", "swim_bouts"}
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
    core_source,
    core_series_picker,
    core_time_window,
    eye_representation_picker,
    load_core_behavior_projection,
    selected_analysis_id,
    selected_eye_run,
    selected_provider,
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
                    selected_eye_run.run_name
                    if selected_eye_run is not None
                    else None
                ),
                eye_representation=(
                    str(eye_representation_picker.value)
                    if eye_representation_picker is not None
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
def _(build_core_behavior_output, core_error, core_projection, go, mo, px):
    if core_error:
        core_output = mo.md(f"Core behavior analysis failed: `{core_error}`")
    elif core_projection is not None:
        core_output = build_core_behavior_output(
            mo, go, px, projection=core_projection
        )
    else:
        core_output = mo.md("")
    core_output
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
        if selected_analysis_id in {"distance", "egocentric_bearing", "polar_distance", "alignment"}:
            items.append(chaser_controls.chaser_picker)
        if selected_analysis_id == "egocentric_bearing":
            items.append(chaser_controls.egocentric_epoch_picker)
        else:
            items.append(chaser_controls.epoch_picker)
            if chaser_controls.epoch_options.get(chaser_controls.epoch_picker.value) is None:
                items.append(chaser_controls.time_window)
        if selected_analysis_id in {"polar_distance", "position_heatmap"}:
            items.append(chaser_controls.heatmap_bins)
        if selected_analysis_id == "position_heatmap":
            items.append(chaser_controls.chaser_overlay)
        if selected_analysis_id == "spatial_occupancy" and chaser_controls.spatial_zone_set_picker is not None:
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
def _(chaser_controls, chaser_loaded, resolve_time_windows_from_multiselect, selected_analysis_id):
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
    build_arena_heatmap,
    build_chaser_gaze_tracking_output,
    build_cra_near_field_output,
    build_cra_primary_endpoint_output,
    build_debug_tables,
    build_detection_occupancy_output,
    build_distance_figure,
    build_egocentric_alignment_output,
    build_egocentric_bearing_output,
    build_egocentric_polar_heatmap_output,
    build_epoch_summary_output,
    build_escape_freeze_output,
    build_fish_heading_output,
    build_spatial_occupancy_output,
    build_spec_provenance_panel,
    build_static_artifacts_panel,
    chaser_controls,
    chaser_egocentric_windows,
    chaser_error,
    chaser_gaze_view,
    chaser_loaded,
    chaser_window,
    go,
    mo,
    px,
    selected_analysis_id,
    selected_provider,
    selected_spec,
    zarr_path,
):
    if selected_provider is None or selected_provider.provider_id != "stimulus_chaser":
        chaser_output = mo.md("")
    elif chaser_error:
        chaser_output = mo.md(f"Chaser analysis failed: `{chaser_error}`")
    elif selected_analysis_id == "static_artifacts" and selected_spec is not None:
        chaser_output = build_static_artifacts_panel(
            mo, zarr_path=zarr_path, run_path=selected_spec.run_path, spec=selected_spec.spec
        )
    elif selected_analysis_id == "provenance" and selected_spec is not None:
        chaser_output = build_spec_provenance_panel(
            mo,
            spec=selected_spec.spec,
            artifact_attrs=selected_spec.attrs,
            option=selected_spec,
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
            mo, go, loaded=chaser_loaded, windows=chaser_egocentric_windows,
            chaser_picker=chaser_controls.chaser_picker,
        )
    elif selected_analysis_id == "polar_distance" and chaser_window is not None:
        chaser_output = build_egocentric_polar_heatmap_output(
            mo, go, loaded=chaser_loaded, window=chaser_window,
            chaser_picker=chaser_controls.chaser_picker,
        )
    elif selected_analysis_id == "fish_heading" and chaser_window is not None:
        chaser_output = build_fish_heading_output(
            mo, go, loaded=chaser_loaded, window=chaser_window
        )
    elif selected_analysis_id == "alignment" and chaser_window is not None:
        chaser_output = build_egocentric_alignment_output(
            mo, go, loaded=chaser_loaded, window=chaser_window,
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
            mo, go, loaded=chaser_loaded, window=chaser_window,
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
            chaser_gaze_view if chaser_gaze_view is not None else chaser_loaded
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
    return (exploration,)


if __name__ == "__main__":
    app.run()
