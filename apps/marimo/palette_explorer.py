#!/usr/bin/env python3
"""Read-only, capability-routed explorer for one Palette recording Zarr."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import time

    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from apps.marimo.components.analysis_catalog import (
        PROVIDERS,
        analyses_for_provider,
        group_specs_by_provider,
    )
    from apps.marimo.components.core_behavior import (
        CoreBehaviorSource,
        build_core_behavior_output,
        load_core_behavior_projection,
    )
    from apps.marimo.components.goodcopbadcop_chaser import (
        available_chaser_analysis_ids,
        build_arena_heatmap,
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
        load_goodcopbadcop_view,
        resolve_time_window_from_widgets,
        resolve_time_windows_from_multiselect,
    )
    from apps.marimo.components.provenance import build_spec_provenance_panel
    from apps.marimo.components.registry import (
        discover_recording_explorer_spec_options,
        discover_protocol_recording_options,
        infer_recordings_root_from_zarr_path,
    )
    from apps.marimo.components.static_artifacts import build_static_artifacts_panel

    # Preserve per-section widget identity as additional lazy sections become
    # visible. The cache lives for the lifetime of this read-only app process.
    chaser_controls_cache = {}

    return (
        PROVIDERS,
        CoreBehaviorSource,
        Path,
        analyses_for_provider,
        available_chaser_analysis_ids,
        build_arena_heatmap,
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
        chaser_controls_cache,
        discover_recording_explorer_spec_options,
        discover_protocol_recording_options,
        go,
        group_specs_by_provider,
        infer_recordings_root_from_zarr_path,
        load_core_behavior_projection,
        load_goodcopbadcop_view,
        mo,
        pd,
        px,
        resolve_time_window_from_widgets,
        resolve_time_windows_from_multiselect,
        time,
    )


@app.cell
def _(Path, discover_protocol_recording_options, mo):
    cli_args = mo.cli_args()
    zarr_path_raw = cli_args.get("zarr-path")
    if not zarr_path_raw:
        raise ValueError(
            "Required --zarr-path is missing. Run: scripts/py -m marimo run "
            "apps/marimo/palette_explorer.py -- --zarr-path <analysis.zarr>"
        )
    seed_zarr_path = Path(str(zarr_path_raw))
    recordings_root_raw = cli_args.get("recordings-root")
    registry_raw = cli_args.get("registry")
    name_contains = cli_args.get("recording-name-contains", "GoodCopBadCop")
    initial_renderer = cli_args.get("renderer")
    initial_run_path = cli_args.get("run-path")
    initial_artifact = cli_args.get("artifact")
    recording_options = discover_protocol_recording_options(
        seed_zarr_path,
        recordings_root=Path(str(recordings_root_raw)) if recordings_root_raw else None,
        registry_path=Path(str(registry_raw)) if registry_raw else None,
        renderer_filter=str(initial_renderer) if initial_renderer else None,
        run_path_filter=str(initial_run_path) if initial_run_path else None,
        artifact_filter=str(initial_artifact) if initial_artifact else None,
        name_contains=str(name_contains) if name_contains else None,
        recording_explorer_only=True,
    )
    if not recording_options:
        raise ValueError(f"No recording Zarrs were discovered from {seed_zarr_path}")
    return (
        initial_artifact,
        initial_renderer,
        initial_run_path,
        recording_options,
        seed_zarr_path,
    )


@app.cell
def _(infer_recordings_root_from_zarr_path, mo, recording_options, seed_zarr_path):
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
    recording_context = mo.vstack(
        [
            recording_picker,
            mo.md(
                f"Search root: `{infer_recordings_root_from_zarr_path(seed_zarr_path)}` · "
                f"{len(recording_options):,} recording(s)"
            ),
        ]
    )
    return recording_by_label, recording_context, recording_picker


@app.cell
def _(
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
    return selected_recording, spec_options, zarr_path


@app.cell
def _(PROVIDERS, group_specs_by_provider, mo, spec_options, zarr_path):
    specs_by_provider = group_specs_by_provider(spec_options)
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
    else:
        provider_picker = None
    return provider_by_label, provider_picker, specs_by_provider


@app.cell
def _(provider_by_label, provider_picker, specs_by_provider):
    if provider_picker is None:
        selected_provider = None
        provider_specs = []
    else:
        selected_provider = provider_by_label[provider_picker.value]
        provider_specs = specs_by_provider[selected_provider.provider_id]
    return provider_specs, selected_provider


@app.cell
def _(mo, provider_specs, selected_provider):
    if selected_provider is None or not provider_specs:
        source_by_label = {}
        source_picker = None
    else:
        source_by_label = {
            f"{index + 1}. {option.run_name or option.run_path}": option
            for index, option in enumerate(provider_specs)
        }
        source_picker = mo.ui.dropdown(
            options=list(source_by_label), value=next(iter(source_by_label)), label="Analysis run"
        )
    return source_by_label, source_picker


@app.cell
def _(source_by_label, source_picker):
    selected_spec = source_by_label[source_picker.value] if source_picker is not None else None
    return (selected_spec,)


@app.cell
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
    return analysis_by_label, core_source


@app.cell
def _(analysis_by_label, mo, selected_provider):
    analysis_by_id = {
        definition.analysis_id: definition for definition in analysis_by_label.values()
    }
    provider_id = selected_provider.provider_id if selected_provider is not None else "analysis"
    section_anchor_by_id = {
        analysis_id: f"{provider_id}-{analysis_id.replace('_', '-')}"
        for analysis_id in analysis_by_id
    }
    # Each sentinel changes value only after it enters the viewport. Downstream
    # projection cells use that value as the read boundary.
    section_lazy_by_id = {
        analysis_id: mo.lazy(
            lambda: mo.md(""),
            show_loading_indicator=True,
        )
        for analysis_id in analysis_by_id
    }
    return analysis_by_id, section_anchor_by_id, section_lazy_by_id


@app.cell
def _(analysis_by_id, core_source, mo, selected_provider):
    core_controls_by_id = {}
    if (
        selected_provider is not None
        and selected_provider.provider_id == "core_behavior"
        and core_source is not None
    ):
        core_start_s, core_stop_s = core_source.time_bounds()
        for _analysis_id in analysis_by_id:
            _series_picker = None
            if _analysis_id in {"speed", "heading"}:
                _series_picker = mo.ui.multiselect(
                    options=list(core_source.series_for(_analysis_id)),
                    value=list(core_source.default_series_for(_analysis_id)),
                    label="Series",
                )
            _time_window = None
            if _analysis_id in {"speed", "heading", "position", "eye_angles", "swim_bouts"}:
                _time_window = mo.ui.range_slider(
                    start=core_start_s,
                    stop=max(core_stop_s, core_start_s + 1e-9),
                    value=(core_start_s, core_stop_s),
                    step=max((core_stop_s - core_start_s) / 1000.0, 0.001),
                    label="Time window (s)",
                    full_width=True,
                )
            core_controls_by_id[_analysis_id] = (_series_picker, _time_window)
    return (core_controls_by_id,)


@app.cell
def _(
    analysis_by_id,
    build_core_behavior_output,
    core_controls_by_id,
    core_source,
    go,
    load_core_behavior_projection,
    mo,
    px,
    section_lazy_by_id,
    selected_provider,
):
    core_control_outputs = {}
    core_outputs = {}
    if selected_provider is not None and selected_provider.provider_id == "core_behavior":
        for _analysis_id in analysis_by_id:
            _series_picker, _time_window = core_controls_by_id.get(_analysis_id, (None, None))
            _control_items = [
                item for item in (_series_picker, _time_window) if item is not None
            ]
            core_control_outputs[_analysis_id] = (
                mo.vstack(_control_items) if _control_items else mo.md("")
            )
            if not section_lazy_by_id[_analysis_id].value or core_source is None:
                core_outputs[_analysis_id] = mo.md("_This analysis loads when it enters view._")
                continue
            try:
                if _time_window is None:
                    _start_s = _stop_s = None
                else:
                    _start_s, _stop_s = _time_window.value
                _projection = load_core_behavior_projection(
                    core_source,
                    _analysis_id,
                    start_s=_start_s,
                    stop_s=_stop_s,
                    series_keys=(
                        tuple(_series_picker.value) if _series_picker is not None else None
                    ),
                )
                core_outputs[_analysis_id] = build_core_behavior_output(
                    mo, go, px, projection=_projection
                )
            except Exception as exc:
                core_outputs[_analysis_id] = mo.md(
                    f"Core behavior analysis failed: `{exc}`"
                )
    return core_control_outputs, core_outputs


@app.cell
def _(
    analysis_by_id,
    load_goodcopbadcop_view,
    section_lazy_by_id,
    selected_provider,
    selected_spec,
    time,
    zarr_path,
):
    chaser_errors_by_id = {}
    chaser_loaded_by_id = {}
    if (
        selected_provider is not None
        and selected_provider.provider_id == "stimulus_chaser"
        and selected_spec is not None
    ):
        for _analysis_id in analysis_by_id:
            if (
                not section_lazy_by_id[_analysis_id].value
                or _analysis_id in {"static_artifacts", "provenance"}
            ):
                continue
            try:
                chaser_loaded_by_id[_analysis_id] = load_goodcopbadcop_view(
                    zarr_path,
                    selected_spec,
                    timer=time,
                    include_companion_analyses=_analysis_id
                    in {"cra_quadrant", "cra_near_field", "escape_freeze"},
                    analysis_id=_analysis_id,
                )
            except Exception as exc:
                chaser_errors_by_id[_analysis_id] = str(exc)
    return chaser_errors_by_id, chaser_loaded_by_id


@app.cell
def _(
    build_chaser_controls,
    chaser_controls_cache,
    chaser_loaded_by_id,
    mo,
    selected_spec,
    zarr_path,
):
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
    chaser_controls_by_id = {}
    for _analysis_id, _loaded in chaser_loaded_by_id.items():
        if _analysis_id not in analyses_with_controls:
            continue
        _cache_key = (
            str(zarr_path),
            selected_spec.artifact_path if selected_spec is not None else "",
            _analysis_id,
        )
        if _cache_key not in chaser_controls_cache:
            chaser_controls_cache[_cache_key] = build_chaser_controls(mo, loaded=_loaded)
        chaser_controls_by_id[_analysis_id] = chaser_controls_cache[_cache_key]
    return (chaser_controls_by_id,)


@app.cell
def _(
    analysis_by_id,
    chaser_controls_by_id,
    chaser_loaded_by_id,
    mo,
    resolve_time_window_from_widgets,
    resolve_time_windows_from_multiselect,
):
    chaser_control_outputs = {}
    chaser_egocentric_windows_by_id = {}
    chaser_windows_by_id = {}
    for _analysis_id in analysis_by_id:
        _controls = chaser_controls_by_id.get(_analysis_id)
        _loaded = chaser_loaded_by_id.get(_analysis_id)
        if _controls is None or _loaded is None:
            chaser_control_outputs[_analysis_id] = mo.md("")
            continue
        _items = []
        if _analysis_id == "distance":
            _items.append(_controls.distance_series_picker)
        if _analysis_id in {"distance", "egocentric_bearing", "polar_distance", "alignment"}:
            _items.append(_controls.chaser_picker)
        if _analysis_id == "egocentric_bearing":
            _items.append(_controls.egocentric_epoch_picker)
            chaser_egocentric_windows_by_id[_analysis_id] = resolve_time_windows_from_multiselect(
                epoch_options=_controls.epoch_options,
                epoch_picker=_controls.egocentric_epoch_picker,
                windows_df=_loaded.windows_df,
            )
        else:
            _items.append(_controls.epoch_picker)
            if _controls.epoch_options.get(_controls.epoch_picker.value) is None:
                _items.append(_controls.time_window)
        if _analysis_id in {"polar_distance", "position_heatmap"}:
            _items.append(_controls.heatmap_bins)
        if _analysis_id == "position_heatmap":
            _items.append(_controls.chaser_overlay)
        if (
            _analysis_id == "spatial_occupancy"
            and _controls.spatial_zone_set_picker is not None
        ):
            _items.append(_controls.spatial_zone_set_picker)
        chaser_control_outputs[_analysis_id] = mo.vstack(_items)
        chaser_windows_by_id[_analysis_id] = resolve_time_window_from_widgets(
            epoch_options=_controls.epoch_options,
            epoch_picker=_controls.epoch_picker,
            time_window=_controls.time_window,
            windows_df=_loaded.windows_df,
        )
    return (
        chaser_control_outputs,
        chaser_egocentric_windows_by_id,
        chaser_windows_by_id,
    )


@app.cell
def _(
    analysis_by_id,
    build_arena_heatmap,
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
    build_spec_provenance_panel,
    build_static_artifacts_panel,
    chaser_controls_by_id,
    chaser_egocentric_windows_by_id,
    chaser_errors_by_id,
    chaser_loaded_by_id,
    chaser_windows_by_id,
    go,
    mo,
    px,
    section_lazy_by_id,
    selected_provider,
    selected_spec,
    zarr_path,
):
    chaser_outputs = {}
    if selected_provider is not None and selected_provider.provider_id == "stimulus_chaser":
        for _analysis_id in analysis_by_id:
            if not section_lazy_by_id[_analysis_id].value:
                chaser_outputs[_analysis_id] = mo.md(
                    "_This analysis loads when it enters view._"
                )
                continue
            if _analysis_id in chaser_errors_by_id:
                chaser_outputs[_analysis_id] = mo.md(
                    f"Chaser analysis failed: `{chaser_errors_by_id[_analysis_id]}`"
                )
                continue
            if _analysis_id == "static_artifacts" and selected_spec is not None:
                chaser_outputs[_analysis_id] = build_static_artifacts_panel(
                    mo,
                    zarr_path=zarr_path,
                    run_path=selected_spec.run_path,
                    spec=selected_spec.spec,
                )
                continue
            if _analysis_id == "provenance" and selected_spec is not None:
                chaser_outputs[_analysis_id] = build_spec_provenance_panel(
                    mo,
                    spec=selected_spec.spec,
                    artifact_attrs=selected_spec.attrs,
                    option=selected_spec,
                )
                continue
            _loaded = chaser_loaded_by_id.get(_analysis_id)
            _controls = chaser_controls_by_id.get(_analysis_id)
            _window = chaser_windows_by_id.get(_analysis_id)
            if _loaded is None:
                chaser_outputs[_analysis_id] = mo.md(
                    "The analysis has no renderable persisted rows."
                )
            elif _analysis_id == "distance" and _controls is not None and _window is not None:
                chaser_outputs[_analysis_id], _ = build_distance_figure(
                    go,
                    loaded=_loaded,
                    distance_series_picker=_controls.distance_series_picker,
                    window=_window,
                )
            elif _analysis_id == "epoch_summary":
                chaser_outputs[_analysis_id] = build_epoch_summary_output(
                    mo, go, loaded=_loaded, chaser_picker=None
                )
            elif _analysis_id == "egocentric_bearing" and _controls is not None:
                chaser_outputs[_analysis_id] = build_egocentric_bearing_output(
                    mo,
                    go,
                    loaded=_loaded,
                    windows=chaser_egocentric_windows_by_id.get(_analysis_id, ()),
                    chaser_picker=_controls.chaser_picker,
                )
            elif _analysis_id == "polar_distance" and _controls is not None and _window is not None:
                chaser_outputs[_analysis_id] = build_egocentric_polar_heatmap_output(
                    mo,
                    go,
                    loaded=_loaded,
                    window=_window,
                    chaser_picker=_controls.chaser_picker,
                )
            elif _analysis_id == "fish_heading" and _window is not None:
                chaser_outputs[_analysis_id] = build_fish_heading_output(
                    mo, go, loaded=_loaded, window=_window
                )
            elif _analysis_id == "alignment" and _controls is not None and _window is not None:
                chaser_outputs[_analysis_id] = build_egocentric_alignment_output(
                    mo,
                    go,
                    loaded=_loaded,
                    window=_window,
                    chaser_picker=_controls.chaser_picker,
                )
            elif _analysis_id == "position_heatmap" and _controls is not None and _window is not None:
                chaser_outputs[_analysis_id], _ = build_arena_heatmap(
                    px,
                    loaded=_loaded,
                    heatmap_bins=_controls.heatmap_bins,
                    chaser_overlay=_controls.chaser_overlay,
                    window=_window,
                )
            elif _analysis_id == "detection_occupancy" and _window is not None:
                chaser_outputs[_analysis_id] = build_detection_occupancy_output(
                    mo, go, loaded=_loaded, window=_window
                )
            elif _analysis_id == "spatial_occupancy" and _controls is not None and _window is not None:
                chaser_outputs[_analysis_id] = build_spatial_occupancy_output(
                    mo,
                    go,
                    loaded=_loaded,
                    window=_window,
                    spatial_zone_set_picker=_controls.spatial_zone_set_picker,
                )
            elif _analysis_id == "cra_quadrant":
                chaser_outputs[_analysis_id] = build_cra_primary_endpoint_output(
                    mo, go, loaded=_loaded
                )
            elif _analysis_id == "cra_near_field":
                chaser_outputs[_analysis_id] = build_cra_near_field_output(
                    mo, go, loaded=_loaded
                )
            elif _analysis_id == "escape_freeze":
                chaser_outputs[_analysis_id] = build_escape_freeze_output(
                    mo, loaded=_loaded
                )
            else:
                chaser_outputs[_analysis_id] = mo.md(
                    "The analysis has no renderable persisted rows."
                )
    return (chaser_outputs,)


@app.cell
def _(
    analysis_by_id,
    chaser_control_outputs,
    chaser_outputs,
    core_control_outputs,
    core_outputs,
    mo,
    provider_picker,
    recording_context,
    section_anchor_by_id,
    section_lazy_by_id,
    selected_provider,
    source_picker,
    zarr_path,
):
    if selected_provider is None:
        navigation = {"#explorer-top": "Overview"}
    else:
        navigation = {"#explorer-top": "Overview"}
        navigation.update(
            {
                f"#{section_anchor_by_id[analysis_id]}": definition.label
                for analysis_id, definition in analysis_by_id.items()
            }
        )
    sidebar_items = [mo.md("## Recording explorer"), recording_context]
    if provider_picker is not None:
        sidebar_items.extend([provider_picker])
    if source_picker is not None:
        sidebar_items.extend([source_picker])
    sidebar_items.extend(
        [mo.md("### Contents"), mo.nav_menu(navigation, orientation="vertical")]
    )
    mo.sidebar(mo.vstack(sidebar_items), width="21rem")

    if not analysis_by_id:
        sections = [
            mo.md(
                "No analysis in this class has all required persisted inputs. "
                "Run or backfill the corresponding visualization contract first."
            )
        ]
    else:
        sections = []
        for _analysis_id, _definition in analysis_by_id.items():
            _controls = (
                core_control_outputs.get(_analysis_id, mo.md(""))
                if selected_provider is not None
                and selected_provider.provider_id == "core_behavior"
                else chaser_control_outputs.get(_analysis_id, mo.md(""))
            )
            _output = (
                core_outputs.get(_analysis_id, mo.md(""))
                if selected_provider is not None
                and selected_provider.provider_id == "core_behavior"
                else chaser_outputs.get(_analysis_id, mo.md(""))
            )
            sections.append(
                mo.vstack(
                    [
                        mo.md(
                            f'<div id="{section_anchor_by_id[_analysis_id]}"></div>\n\n'
                            f"## {_definition.label}\n\n{_definition.description}"
                        ),
                        section_lazy_by_id[_analysis_id],
                        _controls,
                        _output,
                    ]
                ).style(min_height="65vh", scroll_margin_top="1rem")
            )

    mo.vstack(
        [
            mo.md(
                '<div id="explorer-top"></div>\n\n# Palette Recording Explorer\n\n'
                "This viewer opens the recording Zarr read-only. Use the contents "
                "sidebar to move through available analyses. "
                "Each visualization reads its bounded Zarr projection only when "
                "its section enters view."
            ),
            mo.md(f"**{selected_provider.label if selected_provider else 'No analysis class'}**"),
            mo.md(f"`{zarr_path}`"),
            *sections,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
