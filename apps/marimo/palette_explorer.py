#!/usr/bin/env python3
"""General Palette marimo explorer for persisted interactive visualization specs.

Run with:

    scripts/py -m marimo run apps/marimo/palette_explorer.py -- \
      --zarr-path /path/to/archive.zarr
"""

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

    from apps.marimo.components.goodcopbadcop_chaser import (
        build_arena_heatmap,
        build_controls as build_goodcopbadcop_controls,
        build_controls_panel_from_widgets as build_goodcopbadcop_controls_panel_from_widgets,
        build_debug_tables as build_goodcopbadcop_debug_tables,
        build_detection_occupancy_output,
        build_distance_figure,
        build_spatial_occupancy_output,
        build_summary as build_goodcopbadcop_summary,
        is_goodcopbadcop_option,
        load_goodcopbadcop_view,
        resolve_time_window_from_widgets as resolve_goodcopbadcop_time_window_from_widgets,
    )
    from apps.marimo.components.provenance import build_spec_provenance_panel
    from apps.marimo.components.registry import (
        discover_interactive_spec_options,
        group_options_by_renderer,
        renderer_registration_for,
        supported_renderer_ids,
    )
    from apps.marimo.components.static_artifacts import build_static_artifacts_panel

    return (
        Path,
        build_arena_heatmap,
        build_detection_occupancy_output,
        build_distance_figure,
        build_goodcopbadcop_controls_panel_from_widgets,
        build_goodcopbadcop_controls,
        build_goodcopbadcop_debug_tables,
        build_goodcopbadcop_summary,
        build_spatial_occupancy_output,
        build_spec_provenance_panel,
        build_static_artifacts_panel,
        discover_interactive_spec_options,
        go,
        group_options_by_renderer,
        is_goodcopbadcop_option,
        load_goodcopbadcop_view,
        mo,
        pd,
        px,
        renderer_registration_for,
        resolve_goodcopbadcop_time_window_from_widgets,
        supported_renderer_ids,
        time,
    )


@app.cell
def _(Path, discover_interactive_spec_options, mo):
    cli_args = mo.cli_args()
    zarr_path_raw = cli_args.get("zarr-path")
    initial_renderer = cli_args.get("renderer")
    initial_run_path = cli_args.get("run-path")
    initial_artifact = cli_args.get("artifact")
    if not zarr_path_raw:
        raise ValueError(
            "Required CLI arg is missing. Run with: "
            "scripts/py -m marimo run apps/marimo/palette_explorer.py -- "
            "--zarr-path <archive.zarr>"
        )
    zarr_path = Path(str(zarr_path_raw))
    spec_options = discover_interactive_spec_options(
        zarr_path,
        renderer_filter=str(initial_renderer) if initial_renderer else None,
        run_path_filter=str(initial_run_path) if initial_run_path else None,
        artifact_filter=str(initial_artifact) if initial_artifact else None,
    )
    if not spec_options:
        raise ValueError(f"No persisted interactive visualization specs were found in {zarr_path}.")
    return initial_artifact, initial_renderer, initial_run_path, spec_options, zarr_path


@app.cell
def _(group_options_by_renderer, mo, renderer_registration_for, spec_options, supported_renderer_ids, zarr_path):
    renderer_groups = group_options_by_renderer(spec_options)
    supported_ids = set(supported_renderer_ids())
    summary = {
        "zarr_path": str(zarr_path),
        "interactive_specs": len(spec_options),
        "supported_specs": sum(1 for option in spec_options if option.renderer in supported_ids),
        "renderers": {renderer: len(options) for renderer, options in renderer_groups.items()},
        "supported_renderers": list(supported_renderer_ids()),
    }
    spec_label_to_option = {
        f"{index + 1}. {option.label}": option
        for index, option in enumerate(spec_options)
    }
    default_label = next(
        (label for label, option in spec_label_to_option.items() if option.is_supported),
        next(iter(spec_label_to_option)),
    )
    spec_picker = mo.ui.dropdown(
        options=list(spec_label_to_option),
        value=default_label,
        label="Interactive view",
    )
    selected_preview = spec_label_to_option[default_label]
    selected_registration_preview = renderer_registration_for(selected_preview.renderer)
    overview = mo.vstack(
        [
            mo.md(f"# Palette Explorer\n\n`{zarr_path}`"),
            mo.hstack(
                [
                    mo.stat(label="Interactive specs", value=f"{len(spec_options):,}"),
                    mo.stat(label="Supported", value=f"{summary['supported_specs']:,}"),
                    mo.stat(label="Renderer types", value=f"{len(renderer_groups):,}"),
                ]
            ),
            spec_picker,
            mo.md(
                "Selected renderer: "
                f"`{selected_registration_preview.label if selected_registration_preview else selected_preview.renderer}`"
            ),
            mo.tree(summary),
        ]
    )
    overview
    return spec_label_to_option, spec_picker


@app.cell
def _(is_goodcopbadcop_option, renderer_registration_for, spec_label_to_option, spec_picker):
    selected_spec_option = spec_label_to_option[spec_picker.value]
    selected_registration = renderer_registration_for(selected_spec_option.renderer)
    selected_component_key = selected_registration.component_key if selected_registration is not None else ""
    selected_is_goodcopbadcop = is_goodcopbadcop_option(selected_spec_option)
    return selected_component_key, selected_is_goodcopbadcop, selected_registration, selected_spec_option


@app.cell
def _(mo, selected_is_goodcopbadcop, selected_registration, selected_spec_option):
    if selected_is_goodcopbadcop:
        unsupported_output = mo.md("")
    else:
        label = selected_registration.label if selected_registration is not None else selected_spec_option.renderer
        unsupported_output = mo.md(
            "This interactive spec was discovered, but no marimo component is registered for "
            f"`{label}` yet. Use the existing protocol-specific notebook or add a renderer component."
        )
    unsupported_output
    return


@app.cell
def _(load_goodcopbadcop_view, selected_is_goodcopbadcop, selected_spec_option, time, zarr_path):
    if selected_is_goodcopbadcop:
        try:
            gcb_loaded = load_goodcopbadcop_view(zarr_path, selected_spec_option, timer=time)
            gcb_load_error = None
        except Exception as exc:
            gcb_loaded = None
            gcb_load_error = str(exc)
    else:
        gcb_loaded = None
        gcb_load_error = None
    return gcb_load_error, gcb_loaded


@app.cell
def _(build_goodcopbadcop_summary, gcb_load_error, gcb_loaded, mo, selected_spec_option):
    if gcb_load_error:
        gcb_summary_output = mo.md(f"GoodCopBadCop view failed to load: `{gcb_load_error}`")
    elif gcb_loaded is not None:
        gcb_summary_output = build_goodcopbadcop_summary(
            mo,
            loaded=gcb_loaded,
            option=selected_spec_option,
        )
    else:
        gcb_summary_output = mo.md("")
    gcb_summary_output
    return


@app.cell
def _(build_goodcopbadcop_controls, gcb_loaded, mo):
    if gcb_loaded is not None:
        gcb_controls = build_goodcopbadcop_controls(mo, loaded=gcb_loaded)
        gcb_distance_series_picker = gcb_controls.distance_series_picker
        gcb_time_range = gcb_controls.time_window
        gcb_epoch_picker = gcb_controls.epoch_picker
        gcb_epoch_options = gcb_controls.epoch_options
        gcb_heatmap_bins = gcb_controls.heatmap_bins
        gcb_chaser_overlay = gcb_controls.chaser_overlay
        gcb_spatial_zone_set_picker = gcb_controls.spatial_zone_set_picker
    else:
        gcb_controls = None
        gcb_distance_series_picker = None
        gcb_time_range = None
        gcb_epoch_picker = None
        gcb_epoch_options = {}
        gcb_heatmap_bins = None
        gcb_chaser_overlay = None
        gcb_spatial_zone_set_picker = None
    return (
        gcb_chaser_overlay,
        gcb_controls,
        gcb_distance_series_picker,
        gcb_epoch_options,
        gcb_epoch_picker,
        gcb_heatmap_bins,
        gcb_spatial_zone_set_picker,
        gcb_time_range,
    )


@app.cell
def _(
    build_goodcopbadcop_controls_panel_from_widgets,
    gcb_chaser_overlay,
    gcb_distance_series_picker,
    gcb_epoch_options,
    gcb_epoch_picker,
    gcb_heatmap_bins,
    gcb_spatial_zone_set_picker,
    gcb_time_range,
    mo,
):
    if (
        gcb_distance_series_picker is not None
        and gcb_epoch_picker is not None
        and gcb_time_range is not None
        and gcb_heatmap_bins is not None
        and gcb_chaser_overlay is not None
    ):
        gcb_controls_output = build_goodcopbadcop_controls_panel_from_widgets(
            mo,
            distance_series_picker=gcb_distance_series_picker,
            time_window=gcb_time_range,
            epoch_picker=gcb_epoch_picker,
            epoch_options=gcb_epoch_options,
            heatmap_bins=gcb_heatmap_bins,
            chaser_overlay=gcb_chaser_overlay,
            spatial_zone_set_picker=gcb_spatial_zone_set_picker,
        )
    else:
        gcb_controls_output = mo.md("")
    gcb_controls_output
    return


@app.cell
def _(
    gcb_epoch_options,
    gcb_epoch_picker,
    gcb_loaded,
    gcb_time_range,
    resolve_goodcopbadcop_time_window_from_widgets,
):
    if gcb_loaded is not None and gcb_epoch_picker is not None and gcb_time_range is not None:
        gcb_time_window = resolve_goodcopbadcop_time_window_from_widgets(
            epoch_options=gcb_epoch_options,
            epoch_picker=gcb_epoch_picker,
            time_window=gcb_time_range,
            windows_df=gcb_loaded.windows_df,
        )
    else:
        gcb_time_window = None
    return (gcb_time_window,)


@app.cell
def _(build_distance_figure, gcb_distance_series_picker, gcb_loaded, gcb_time_window, go, mo, pd):
    if gcb_loaded is not None and gcb_distance_series_picker is not None and gcb_time_window is not None:
        gcb_distance_output, gcb_visible_distance_df = build_distance_figure(
            go,
            loaded=gcb_loaded,
            distance_series_picker=gcb_distance_series_picker,
            window=gcb_time_window,
        )
    else:
        gcb_distance_output = mo.md("")
        gcb_visible_distance_df = pd.DataFrame()
    gcb_distance_output
    return (gcb_visible_distance_df,)


@app.cell
def _(
    build_arena_heatmap,
    gcb_chaser_overlay,
    gcb_heatmap_bins,
    gcb_loaded,
    gcb_time_window,
    mo,
    pd,
    px,
):
    if (
        gcb_loaded is not None
        and gcb_chaser_overlay is not None
        and gcb_heatmap_bins is not None
        and gcb_time_window is not None
    ):
        gcb_arena_output, gcb_visible_position_df = build_arena_heatmap(
            px,
            loaded=gcb_loaded,
            heatmap_bins=gcb_heatmap_bins,
            chaser_overlay=gcb_chaser_overlay,
            window=gcb_time_window,
        )
    else:
        gcb_arena_output = mo.md("")
        gcb_visible_position_df = pd.DataFrame()
    gcb_arena_output
    return (gcb_visible_position_df,)


@app.cell
def _(build_detection_occupancy_output, gcb_loaded, gcb_time_window, go, mo):
    if gcb_loaded is not None and gcb_time_window is not None:
        gcb_occupancy_output = build_detection_occupancy_output(
            mo,
            go,
            loaded=gcb_loaded,
            window=gcb_time_window,
        )
    else:
        gcb_occupancy_output = mo.md("")
    gcb_occupancy_output
    return


@app.cell
def _(build_spatial_occupancy_output, gcb_loaded, gcb_spatial_zone_set_picker, gcb_time_window, go, mo):
    if gcb_loaded is not None and gcb_time_window is not None:
        gcb_spatial_occupancy_output = build_spatial_occupancy_output(
            mo,
            go,
            loaded=gcb_loaded,
            window=gcb_time_window,
            spatial_zone_set_picker=gcb_spatial_zone_set_picker,
        )
    else:
        gcb_spatial_occupancy_output = mo.md("")
    gcb_spatial_occupancy_output
    return


@app.cell
def _(build_static_artifacts_panel, gcb_loaded, mo, selected_spec_option, zarr_path):
    if gcb_loaded is not None:
        static_artifacts_output = build_static_artifacts_panel(
            mo,
            zarr_path=zarr_path,
            run_path=selected_spec_option.run_path,
            spec=gcb_loaded.data.spec,
        )
    else:
        static_artifacts_output = mo.md("")
    static_artifacts_output
    return


@app.cell
def _(
    build_goodcopbadcop_debug_tables,
    build_spec_provenance_panel,
    gcb_loaded,
    gcb_visible_distance_df,
    gcb_visible_position_df,
    mo,
    selected_spec_option,
):
    if gcb_loaded is not None:
        debug_tables = build_goodcopbadcop_debug_tables(
            mo,
            loaded=gcb_loaded,
            visible_distance_df=gcb_visible_distance_df,
            visible_position_df=gcb_visible_position_df,
        )
        provenance = build_spec_provenance_panel(
            mo,
            spec=gcb_loaded.data.spec,
            artifact_attrs=gcb_loaded.data.attrs,
            option=selected_spec_option,
        )
        detail_output = mo.vstack([debug_tables, provenance])
    else:
        detail_output = mo.md("")
    detail_output
    return


if __name__ == "__main__":
    app.run()
