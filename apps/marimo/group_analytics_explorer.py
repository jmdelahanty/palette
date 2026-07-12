#!/usr/bin/env python3
"""Read-only Marimo explorer for immutable Palette analytics exports.

Run with:

    scripts/py -m marimo run apps/marimo/group_analytics_explorer.py -- \
      --export-root /groups/johnson/johnsonlab/palette_analytics \
      --export-run-id latest \
      --stats-run-id auto
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from apps.marimo.components.group_analytics import (
        available_group_panels,
        chaser_selection_options,
        egocentric_heatmap_figure,
        filter_rows_by_chasers,
        grouped_bar_figure,
        line_figure,
        panel_control_spec,
        sample_grain_status_rows,
    )
    from fisheye.group_analytics_viewer.catalog import (
        discover_export_catalog,
        select_export_run_id,
    )
    from fisheye.group_analytics_viewer.query import (
        build_context,
        build_health_report,
        query_chaser_histogram,
        query_chaser_summary,
        query_cra_near_field_curves,
        query_cra_near_field_object_phase,
        query_cra_near_field_summary,
        query_cra_object_phase,
        query_cra_summary,
        query_egocentric_histogram,
        query_egocentric_summary,
        query_epoch_bout_histogram,
        query_epoch_inter_bout_interval_histogram,
        query_epoch_speed_summary,
        query_export_summary,
        query_group_statistics,
        query_options,
        query_recordings,
        query_spatial_occupancy,
        query_speed_distance_bins,
    )

    return (
        Path,
        available_group_panels,
        build_context,
        build_health_report,
        chaser_selection_options,
        discover_export_catalog,
        egocentric_heatmap_figure,
        filter_rows_by_chasers,
        grouped_bar_figure,
        line_figure,
        mo,
        os,
        panel_control_spec,
        pd,
        query_chaser_histogram,
        query_chaser_summary,
        query_cra_near_field_curves,
        query_cra_near_field_object_phase,
        query_cra_near_field_summary,
        query_cra_object_phase,
        query_cra_summary,
        query_egocentric_histogram,
        query_egocentric_summary,
        query_epoch_bout_histogram,
        query_epoch_inter_bout_interval_histogram,
        query_epoch_speed_summary,
        query_export_summary,
        query_group_statistics,
        query_options,
        query_recordings,
        query_spatial_occupancy,
        query_speed_distance_bins,
        sample_grain_status_rows,
        select_export_run_id,
    )


@app.cell
def _(Path, discover_export_catalog, mo, os, pd, select_export_run_id):
    cli_args = mo.cli_args()
    default_export_root = os.environ.get(
        "PALETTE_ANALYTICS_EXPORT_ROOT",
        "/groups/johnson/johnsonlab/palette_analytics",
    )
    export_root = Path(str(cli_args.get("export-root", default_export_root)))
    requested_export_run_id = str(cli_args.get("export-run-id", "latest"))
    requested_panel_id = str(cli_args.get("panel", "behavior"))
    stats_run_id_raw = cli_args.get("stats-run-id", "auto")
    stats_run_id = (
        None
        if stats_run_id_raw in (None, "", "none", "None")
        else str(stats_run_id_raw)
    )

    export_catalog = discover_export_catalog(export_root)
    catalog_diagnostics = pd.DataFrame(
        [diagnostic.to_dict() for diagnostic in export_catalog.diagnostics]
    )
    if not export_catalog.entries:
        mo.stop(
            True,
            mo.vstack(
                [
                    mo.md("# Palette Group Analytics"),
                    mo.callout(
                        mo.md(
                            f"No selectable V2 analytics exports were found under "
                            f"`{export_catalog.export_root}`."
                        ),
                        kind="warn",
                    ),
                    (
                        mo.ui.table(catalog_diagnostics, selection=None, page_size=10)
                        if not catalog_diagnostics.empty
                        else mo.md("No manifest diagnostics were produced.")
                    ),
                ]
            ),
        )

    selection_warning = None
    try:
        initial_export_run_id = select_export_run_id(
            export_catalog,
            requested_export_run_id,
        )
    except ValueError as exc:
        initial_export_run_id = select_export_run_id(export_catalog, "latest")
        selection_warning = str(exc)

    label_to_export_run_id = {
        entry.label: entry.export_run_id for entry in export_catalog.entries
    }
    initial_label = next(
        label
        for label, run_id in label_to_export_run_id.items()
        if run_id == initial_export_run_id
    )
    export_picker = mo.ui.dropdown(
        options=list(label_to_export_run_id),
        value=initial_label,
        label="Export dataset",
        searchable=True,
    )
    selector_parts = [
        mo.md("# Palette Group Analytics"),
        mo.md(f"Authorized export root: `{export_catalog.export_root}`"),
        export_picker,
    ]
    if selection_warning:
        selector_parts.append(
            mo.callout(
                mo.md(f"{selection_warning} The newest export was selected instead."),
                kind="warn",
            )
        )
    selector_parts.append(
        mo.accordion(
            {
                "Available exports": mo.ui.table(
                    pd.DataFrame([entry.to_dict() for entry in export_catalog.entries]),
                    selection=None,
                    page_size=10,
                ),
                "Rejected manifests": (
                    mo.ui.table(catalog_diagnostics, selection=None, page_size=10)
                    if not catalog_diagnostics.empty
                    else mo.md("No manifests were rejected.")
                ),
            }
        )
    )
    mo.vstack(selector_parts)
    return (
        export_catalog,
        export_picker,
        export_root,
        label_to_export_run_id,
        requested_panel_id,
        stats_run_id,
    )


@app.cell
def _(
    build_context,
    export_picker,
    export_root,
    label_to_export_run_id,
    stats_run_id,
):
    selected_export_run_id = label_to_export_run_id[export_picker.value]
    context = build_context(
        export_root=export_root,
        export_run_id=selected_export_run_id,
        stats_run_id=stats_run_id,
    )
    return context, selected_export_run_id


@app.cell
def _(
    available_group_panels,
    build_health_report,
    context,
    export_catalog,
    mo,
    pd,
    query_export_summary,
    sample_grain_status_rows,
    selected_export_run_id,
):
    health = build_health_report(context)
    summary = query_export_summary(context)
    selected_entry = export_catalog.entry(selected_export_run_id)
    collection = summary.get("collection") or {}
    source_recording_count = summary.get("source_recording_count")
    selected_capabilities = set(selected_entry.capabilities if selected_entry else ())
    if summary["statistics"].get("available"):
        selected_capabilities.add("group.statistics")
        if summary["statistics"].get("descriptive_row_count") is not None:
            selected_capabilities.add("group.descriptive_statistics")
    panel_definitions = available_group_panels(
        selected_capabilities,
        statistics_available=bool(summary["statistics"].get("available")),
    )
    sample_grain_rows = sample_grain_status_rows(selected_capabilities)

    mo.vstack(
        [
            mo.md(f"## Dataset overview\n\nManifest: `{summary['manifest_path']}`"),
            mo.hstack(
                [
                    mo.stat(label="Export run", value=summary["export_run_id"]),
                    mo.stat(
                        label="Recordings",
                        value=(
                            f"{source_recording_count:,}"
                            if source_recording_count is not None
                            else "unknown"
                        ),
                    ),
                    mo.stat(label="Tables", value=f"{len(selected_entry.table_names):,}"),
                    mo.stat(
                        label="Stats rows",
                        value=(
                            f"{summary['statistics']['row_count']:,}"
                            if summary["statistics"].get("row_count") is not None
                            else "none"
                        ),
                    ),
                    mo.stat(label="Health", value="ok" if health.ok else "check"),
                ]
            ),
            mo.md(f"Collection: `{collection.get('collection_id', 'none')}`"),
            mo.accordion(
                {
                    "Dense/sample-grain availability": mo.vstack(
                        [
                            mo.md(
                                "These statuses describe the selected immutable export only; "
                                "they do not assert that a source-Zarr analysis is unsupported "
                                "or absent."
                            ),
                            mo.ui.table(
                                pd.DataFrame(sample_grain_rows),
                                selection=None,
                                page_size=8,
                            ),
                        ]
                    )
                }
            ),
        ]
    )
    mo.stop(
        selected_entry is not None and not selected_entry.ready,
        mo.callout(
            mo.md(
                "This manifest references missing Parquet parts. Analysis panels are disabled "
                "so a partial export cannot be presented as complete."
            ),
            kind="warn",
        ),
    )
    analysis_context = context
    return (
        analysis_context,
        health,
        panel_definitions,
        sample_grain_rows,
        selected_capabilities,
        summary,
    )


@app.cell
def _(
    analysis_context,
    chaser_selection_options,
    mo,
    panel_control_spec,
    panel_definitions,
    query_options,
    requested_panel_id,
):
    options = query_options(analysis_context)
    panel_labels = {definition.label: definition.panel_id for definition in panel_definitions}
    initial_panel_label = next(
        (
            label
            for label, panel_id in panel_labels.items()
            if panel_id == requested_panel_id
        ),
        next(iter(panel_labels)),
    )
    panel_picker = mo.ui.dropdown(
        options=list(panel_labels),
        value=initial_panel_label,
        label="Visualization class",
    )
    window_labels = ["All epochs"] + [
        item["window_label"] for item in options.get("windows", [])
    ]
    window_picker = mo.ui.dropdown(
        options=window_labels,
        value=window_labels[0],
        label="Epoch",
    )
    chaser_labels, default_chaser_labels = chaser_selection_options(
        options.get("chasers", [])
    )
    chaser_picker = mo.ui.multiselect(
        options=list(chaser_labels),
        value=default_chaser_labels,
        label="Chasers",
    )
    stat_labels = {"Mean": "mean", "Median": "median"}
    stat_picker = mo.ui.dropdown(
        options=list(stat_labels),
        value="Mean",
        label="Summary statistic",
    )

    def _metric_control(items, *, label, preferred):
        mapping = {
            f"{item['label']} [{item['metric']}]": item["metric"] for item in items
        }
        selected = next(
            (key for key, value in mapping.items() if value == preferred),
            next(iter(mapping)),
        )
        return mapping, mo.ui.dropdown(
            options=list(mapping),
            value=selected,
            label=label,
        )

    _behavior_spec = panel_control_spec("behavior")
    behavior_metric_labels, behavior_metric_picker = _metric_control(
        options[_behavior_spec.analysis_options_key],
        label=_behavior_spec.analysis_label,
        preferred=_behavior_spec.preferred_analysis,
    )
    _bout_spec = panel_control_spec("bout_distributions")
    bout_metric_labels, bout_metric_picker = _metric_control(
        options[_bout_spec.analysis_options_key],
        label=_bout_spec.analysis_label,
        preferred=_bout_spec.preferred_analysis,
    )
    _spatial_spec = panel_control_spec("spatial")
    spatial_metric_labels, spatial_metric_picker = _metric_control(
        options[_spatial_spec.analysis_options_key],
        label=_spatial_spec.analysis_label,
        preferred=_spatial_spec.preferred_analysis,
    )
    _chaser_spec = panel_control_spec("chaser_distance")
    chaser_metric_labels, chaser_metric_picker = _metric_control(
        options[_chaser_spec.analysis_options_key],
        label=_chaser_spec.analysis_label,
        preferred=_chaser_spec.preferred_analysis,
    )
    _cra_spec = panel_control_spec("cra")
    cra_metric_labels, cra_metric_picker = _metric_control(
        options[_cra_spec.analysis_options_key],
        label=_cra_spec.analysis_label,
        preferred=_cra_spec.preferred_analysis,
    )
    _near_spec = panel_control_spec("near_field")
    near_metric_labels, near_metric_picker = _metric_control(
        options[_near_spec.analysis_options_key],
        label=_near_spec.analysis_label,
        preferred=_near_spec.preferred_analysis,
    )
    _egocentric_spec = panel_control_spec("egocentric")
    egocentric_metric_labels, egocentric_metric_picker = _metric_control(
        options[_egocentric_spec.analysis_options_key],
        label=_egocentric_spec.analysis_label,
        preferred=_egocentric_spec.preferred_analysis,
    )
    panel_picker
    return (
        behavior_metric_labels,
        behavior_metric_picker,
        bout_metric_labels,
        bout_metric_picker,
        chaser_labels,
        chaser_metric_labels,
        chaser_metric_picker,
        chaser_picker,
        cra_metric_labels,
        cra_metric_picker,
        egocentric_metric_labels,
        egocentric_metric_picker,
        near_metric_labels,
        near_metric_picker,
        panel_labels,
        panel_picker,
        spatial_metric_labels,
        spatial_metric_picker,
        stat_labels,
        stat_picker,
        window_picker,
    )


@app.cell
def _(
    behavior_metric_picker,
    bout_metric_picker,
    chaser_metric_picker,
    chaser_picker,
    cra_metric_picker,
    egocentric_metric_picker,
    mo,
    near_metric_picker,
    panel_control_spec,
    panel_labels,
    panel_picker,
    spatial_metric_picker,
    stat_picker,
    window_picker,
):
    selected_panel_id = panel_labels[panel_picker.value]
    control_spec = panel_control_spec(selected_panel_id)
    _analysis_picker_by_options_key = {
        "epoch_speed_metrics": behavior_metric_picker,
        "epoch_bout_histogram_metrics": bout_metric_picker,
        "spatial_metrics": spatial_metric_picker,
        "chaser_metrics": chaser_metric_picker,
        "cra_object_phase_metrics": cra_metric_picker,
        "cra_near_field_object_phase_metrics": near_metric_picker,
        "egocentric_metrics": egocentric_metric_picker,
    }
    _controls = []
    _analysis_picker = _analysis_picker_by_options_key.get(
        control_spec.analysis_options_key
    )
    if _analysis_picker is not None:
        _controls.append(_analysis_picker)
    if control_spec.show_window:
        _controls.append(window_picker)
    if control_spec.show_chaser:
        _controls.append(chaser_picker)
    if control_spec.show_statistic:
        _controls.append(stat_picker)
    mo.hstack(_controls) if _controls else mo.md(
        f"**{panel_picker.value}** has no additional analysis controls."
    )
    return (selected_panel_id,)


@app.cell
def _(
    analysis_context,
    behavior_metric_labels,
    behavior_metric_picker,
    bout_metric_labels,
    bout_metric_picker,
    chaser_labels,
    chaser_metric_labels,
    chaser_metric_picker,
    chaser_picker,
    cra_metric_labels,
    cra_metric_picker,
    egocentric_metric_labels,
    egocentric_metric_picker,
    filter_rows_by_chasers,
    near_metric_labels,
    near_metric_picker,
    query_chaser_histogram,
    query_chaser_summary,
    query_cra_near_field_curves,
    query_cra_near_field_object_phase,
    query_cra_near_field_summary,
    query_cra_object_phase,
    query_cra_summary,
    query_egocentric_histogram,
    query_egocentric_summary,
    query_epoch_bout_histogram,
    query_epoch_inter_bout_interval_histogram,
    query_epoch_speed_summary,
    query_group_statistics,
    query_recordings,
    query_spatial_occupancy,
    query_speed_distance_bins,
    selected_capabilities,
    selected_panel_id,
    spatial_metric_labels,
    spatial_metric_picker,
    stat_labels,
    stat_picker,
    window_picker,
):
    selected_window = (
        None if window_picker.value == "All epochs" else str(window_picker.value)
    )
    selected_chasers = tuple(
        chaser_labels[label]
        for label in chaser_picker.value
        if label in chaser_labels
    )
    selected_stat = stat_labels[stat_picker.value]
    payloads = {}
    if (
        selected_panel_id == "behavior"
        and "chaser.epoch.behavior_summary" in selected_capabilities
    ):
        payloads["behavior"] = query_epoch_speed_summary(
            analysis_context,
            metric=behavior_metric_labels[behavior_metric_picker.value],
            stat=selected_stat,
        )
    if (
        selected_panel_id == "bout_distributions"
        and "chaser.epoch.bout_histogram" in selected_capabilities
    ):
        payloads["bout_histogram"] = query_epoch_bout_histogram(
            analysis_context,
            metric=bout_metric_labels[bout_metric_picker.value],
            window_label=selected_window,
        )
    if (
        selected_panel_id == "bout_distributions"
        and "chaser.epoch.inter_bout_interval_histogram" in selected_capabilities
    ):
        payloads["ibi_histogram"] = query_epoch_inter_bout_interval_histogram(
            analysis_context,
            window_label=selected_window,
        )
    if (
        selected_panel_id == "spatial"
        and "chaser.epoch.spatial_occupancy" in selected_capabilities
    ):
        payloads["spatial"] = query_spatial_occupancy(
            analysis_context,
            metric=spatial_metric_labels[spatial_metric_picker.value],
        )
    if (
        selected_panel_id == "chaser_distance"
        and "chaser.distance.summary" in selected_capabilities
    ):
        _chaser_summary = query_chaser_summary(
            analysis_context,
            metric=chaser_metric_labels[chaser_metric_picker.value],
            stat=selected_stat,
        )
        _chaser_summary["rows"] = [
            row
            for row in filter_rows_by_chasers(
                _chaser_summary.get("rows", []),
                selected_chasers,
            )
            if (selected_window is None or row.get("window_label") == selected_window)
        ]
        payloads["chaser_summary"] = _chaser_summary
    if (
        selected_panel_id == "chaser_distance"
        and "chaser.distance.histogram" in selected_capabilities
    ):
        _chaser_histogram = query_chaser_histogram(
            analysis_context,
            window_label=selected_window,
            chaser_index=None,
        )
        _chaser_histogram["rows"] = filter_rows_by_chasers(
            _chaser_histogram.get("rows", []),
            selected_chasers,
        )
        payloads["chaser_histogram"] = _chaser_histogram
    if (
        selected_panel_id == "chaser_distance"
        and "chaser.distance.speed_relationship" in selected_capabilities
    ):
        _speed_distance = query_speed_distance_bins(
            analysis_context,
            window_label=selected_window,
            chaser_index=None,
        )
        _speed_distance["rows"] = filter_rows_by_chasers(
            _speed_distance.get("rows", []),
            selected_chasers,
        )
        payloads["speed_distance"] = _speed_distance
    if selected_panel_id == "cra" and "chaser.cra.primary" in selected_capabilities:
        payloads["cra_phase"] = query_cra_object_phase(
            analysis_context,
            metric=cra_metric_labels[cra_metric_picker.value],
            stat=selected_stat,
        )
        payloads["cra_summary"] = query_cra_summary(analysis_context)
    if (
        selected_panel_id == "near_field"
        and "chaser.cra.near_field" in selected_capabilities
    ):
        payloads["near_phase"] = query_cra_near_field_object_phase(
            analysis_context,
            metric=near_metric_labels[near_metric_picker.value],
            stat=selected_stat,
        )
        payloads["near_summary"] = query_cra_near_field_summary(analysis_context)
        payloads["near_curves"] = query_cra_near_field_curves(analysis_context)
    if (
        selected_panel_id == "egocentric"
        and "chaser.egocentric" in selected_capabilities
    ):
        _egocentric_summary = query_egocentric_summary(
            analysis_context,
            metric=egocentric_metric_labels[egocentric_metric_picker.value],
            stat=selected_stat,
        )
        _egocentric_summary["rows"] = [
            row
            for row in filter_rows_by_chasers(
                _egocentric_summary.get("rows", []),
                selected_chasers,
            )
            if (selected_window is None or row.get("window_label") == selected_window)
        ]
        payloads["egocentric_summary"] = _egocentric_summary
        if selected_window is not None and len(selected_chasers) == 1:
            payloads["egocentric_histogram"] = query_egocentric_histogram(
                analysis_context,
                window_label=selected_window,
                chaser_index=selected_chasers[0],
            )
    if selected_panel_id == "statistics" and "group.statistics" in selected_capabilities:
        payloads["statistics"] = query_group_statistics(analysis_context)
    if selected_panel_id == "inventory":
        payloads["recordings"] = query_recordings(analysis_context)
    return payloads, selected_chasers, selected_window


@app.cell
def _(
    egocentric_heatmap_figure,
    grouped_bar_figure,
    health,
    line_figure,
    mo,
    panel_labels,
    panel_picker,
    payloads,
    pd,
    sample_grain_rows,
    selected_chasers,
    selected_window,
    summary,
):
    selected_panel = panel_labels[panel_picker.value]

    def _rows_table(rows, page_size=12):
        frame = pd.DataFrame(rows)
        return (
            mo.ui.table(frame, selection=None, page_size=page_size)
            if not frame.empty
            else mo.md("No rows are available for the selected filters.")
        )

    def _figure_or_message(figure, message="No plottable rows are available."):
        return figure if figure is not None else mo.md(message)

    if selected_panel == "behavior":
        _data = payloads.get("behavior", {})
        _figure = grouped_bar_figure(
            _data.get("rows", []),
            title=f"Epoch behavior · {_data.get('metric_label', 'metric')}",
            x_key="window_label",
            y_key="value",
            series_key="stat",
            yaxis_title=_data.get("metric_label", "Value"),
        )
        panel_output = mo.vstack(
            [mo.md("## Core behavior"), _figure_or_message(_figure), _rows_table(_data.get("rows", []))]
        )
    elif selected_panel == "bout_distributions":
        _bout = payloads.get("bout_histogram", {})
        _ibi = payloads.get("ibi_histogram", {})
        _bout_figure = line_figure(
            _bout.get("rows", []),
            title=f"Bout distribution · {_bout.get('metric_label', 'metric')}",
            x_key="bin_center",
            y_key="pooled_density",
            series_keys=("window_label",),
            xaxis_title=_bout.get("metric_label", "Bout metric"),
            yaxis_title="Pooled density",
        )
        _ibi_figure = line_figure(
            _ibi.get("rows", []),
            title="Inter-bout interval distribution",
            x_key="bin_center",
            y_key="pooled_density",
            series_keys=("window_label",),
            xaxis_title="Inter-bout interval (s)",
            yaxis_title="Pooled density",
        )
        panel_output = mo.vstack(
            [
                mo.md("## Bout distributions"),
                mo.hstack([_figure_or_message(_bout_figure), _figure_or_message(_ibi_figure)]),
                mo.accordion(
                    {
                        "Bout bins": _rows_table(_bout.get("rows", [])),
                        "Inter-bout interval bins": _rows_table(_ibi.get("rows", [])),
                    }
                ),
            ]
        )
    elif selected_panel == "spatial":
        _data = payloads.get("spatial", {})
        _figure = grouped_bar_figure(
            _data.get("rows", []),
            title=f"Spatial occupancy · {_data.get('metric_label', 'metric')}",
            x_key="window_label",
            y_key="value",
            series_key="zone_label",
            yaxis_title=_data.get("metric_label", "Value"),
        )
        panel_output = mo.vstack(
            [mo.md("## Spatial occupancy"), _figure_or_message(_figure), _rows_table(_data.get("rows", []))]
        )
    elif selected_panel == "chaser_distance":
        _summary_data = payloads.get("chaser_summary", {})
        _summary_rows = [
            {**row, "series": f"{row.get('behavior_class', 'unknown')} · chaser {row.get('chaser_index')}"}
            for row in _summary_data.get("rows", [])
        ]
        _summary_figure = grouped_bar_figure(
            _summary_rows,
            title=f"Chaser distance · {_summary_data.get('metric_label', 'metric')}",
            x_key="window_label",
            y_key="value",
            series_key="series",
            yaxis_title=_summary_data.get("metric_label", "Value"),
            color_key="raw_color_hex",
        )
        _hist = payloads.get("chaser_histogram", {})
        _hist_figure = line_figure(
            _hist.get("rows", []),
            title="Distance distribution",
            x_key="bin_center_mm",
            y_key="pooled_density",
            series_keys=("window_label", "chaser_index"),
            xaxis_title="Distance to chaser (mm)",
            yaxis_title="Pooled density",
            color_key="raw_color_hex",
        )
        _speed = payloads.get("speed_distance", {})
        _speed_figure = line_figure(
            _speed.get("rows", []),
            title="Speed versus chaser distance",
            x_key="distance_bin_center_mm",
            y_key="pooled_mean_speed_mm_s",
            series_keys=("window_label", "chaser_index"),
            xaxis_title="Distance to chaser (mm)",
            yaxis_title="Pooled mean speed (mm/s)",
            color_key="raw_color_hex",
        )
        panel_output = mo.vstack(
            [
                mo.md("## Chaser distance"),
                _figure_or_message(_summary_figure),
                mo.hstack([_figure_or_message(_hist_figure), _figure_or_message(_speed_figure)]),
                mo.accordion(
                    {
                        "Distance summary": _rows_table(_summary_rows),
                        "Distance bins": _rows_table(_hist.get("rows", [])),
                        "Speed-distance bins": _rows_table(_speed.get("rows", [])),
                    }
                ),
            ]
        )
    elif selected_panel == "cra":
        _phase = payloads.get("cra_phase", {})
        _figure = grouped_bar_figure(
            _phase.get("rows", []),
            title=f"CRA object phases · {_phase.get('metric_label', 'metric')}",
            x_key="phase_label",
            y_key="value",
            series_key="object_role",
            yaxis_title=_phase.get("metric_label", "Value"),
            color_key="raw_color_hex",
        )
        _summary_data = payloads.get("cra_summary", {})
        panel_output = mo.vstack(
            [
                mo.md("## CRA primary endpoints"),
                _figure_or_message(_figure),
                mo.accordion(
                    {
                        "Endpoint summaries": _rows_table(_summary_data.get("metrics", [])),
                        "Per-recording endpoints": _rows_table(_summary_data.get("rows", [])),
                        "Object phases": _rows_table(_phase.get("rows", [])),
                    }
                ),
            ]
        )
    elif selected_panel == "near_field":
        _phase = payloads.get("near_phase", {})
        _phase_figure = grouped_bar_figure(
            _phase.get("rows", []),
            title=f"CRA near field · {_phase.get('metric_label', 'metric')}",
            x_key="phase_label",
            y_key="value",
            series_key="object_role",
            yaxis_title=_phase.get("metric_label", "Value"),
            color_key="raw_color_hex",
        )
        _curves = payloads.get("near_curves", {})
        _radial_figure = line_figure(
            _curves.get("radial_rows", []),
            title="Near-field radial density",
            x_key="radial_bin_center_mm",
            y_key="mean",
            series_keys=("phase_label", "object_role"),
            xaxis_title="Radial distance (mm)",
            yaxis_title="Mean density (/mm²)",
        )
        _cdf_figure = line_figure(
            _curves.get("cdf_rows", []),
            title="Near-field distance CDF",
            x_key="distance_threshold_mm",
            y_key="mean",
            series_keys=("phase_label", "object_role"),
            xaxis_title="Distance threshold (mm)",
            yaxis_title="Mean cumulative fraction",
        )
        _summary_data = payloads.get("near_summary", {})
        panel_output = mo.vstack(
            [
                mo.md("## CRA near field"),
                _figure_or_message(_phase_figure),
                mo.hstack([_figure_or_message(_radial_figure), _figure_or_message(_cdf_figure)]),
                mo.accordion(
                    {
                        "Endpoint summaries": _rows_table(_summary_data.get("metrics", [])),
                        "Per-recording endpoints": _rows_table(_summary_data.get("rows", [])),
                        "Object phases": _rows_table(_phase.get("rows", [])),
                    }
                ),
            ]
        )
    elif selected_panel == "egocentric":
        _summary_data = payloads.get("egocentric_summary", {})
        _summary_rows = [
            {**row, "series": f"{row.get('behavior_class', 'unknown')} · chaser {row.get('chaser_index')}"}
            for row in _summary_data.get("rows", [])
        ]
        _summary_figure = grouped_bar_figure(
            _summary_rows,
            title=f"Egocentric bearing · {_summary_data.get('metric_label', 'metric')}",
            x_key="window_label",
            y_key="value",
            series_key="series",
            yaxis_title=_summary_data.get("metric_label", "Value"),
            color_key="raw_color_hex",
        )
        _hist = payloads.get("egocentric_histogram", {})
        _heatmap = egocentric_heatmap_figure(
            _hist.get("rows", []),
            title=f"Distance × bearing · {selected_window}",
        )
        _heatmap_message = (
            "Select one epoch and exactly one chaser to avoid pooling distinct conditions or objects "
            "in the distance-by-bearing heatmap."
            if selected_window is None or len(selected_chasers) != 1
            else "No egocentric histogram rows are available for these filters."
        )
        panel_output = mo.vstack(
            [
                mo.md("## Egocentric bearing"),
                _figure_or_message(_summary_figure),
                _figure_or_message(_heatmap, _heatmap_message),
                mo.accordion(
                    {
                        "Summary rows": _rows_table(_summary_rows),
                        "Histogram bins": _rows_table(_hist.get("rows", [])),
                    }
                ),
            ]
        )
    elif selected_panel == "statistics":
        _stats = payloads.get("statistics", {})
        panel_output = mo.vstack(
            [
                mo.md("## Linked statistics"),
                mo.md("Rows come from the statistics export linked to this immutable base export."),
                _rows_table(_stats.get("rows", [])),
            ]
        )
    else:
        _recordings = payloads.get("recordings", {})
        panel_output = mo.vstack(
            [
                mo.md("## Recordings and provenance"),
                mo.accordion(
                    {
                        "Recordings": _rows_table(_recordings.get("rows", [])),
                        "Export tables": _rows_table(summary.get("tables", []), 18),
                        "Dense/sample-grain availability": _rows_table(sample_grain_rows),
                        "Health details": _rows_table([health.to_dict()]),
                    }
                ),
            ]
        )
    panel_output
    return


if __name__ == "__main__":
    app.run()
