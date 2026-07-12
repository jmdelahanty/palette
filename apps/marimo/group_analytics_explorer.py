#!/usr/bin/env python3
"""Marimo app for Palette group analytics exports.

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
    import plotly.graph_objects as go

    from fisheye.group_analytics_viewer.catalog import (
        discover_export_catalog,
        select_export_run_id,
    )
    from fisheye.group_analytics_viewer.query import (
        CHASER_SUMMARY_TABLE,
        SPATIAL_TABLE,
        build_context,
        build_health_report,
        query_epoch_bout_histogram,
        query_epoch_inter_bout_interval_histogram,
        query_epoch_speed_summary,
        query_export_summary,
        query_group_statistics,
        query_options,
        query_recordings,
    )

    def build_histogram_figure(rows, *, title, xaxis_title):
        frame = pd.DataFrame(rows)
        if frame.empty:
            return None
        fig = go.Figure()
        for label, group in frame.groupby("window_label", sort=False):
            group = group.sort_values("bin_center")
            fig.add_trace(
                go.Bar(
                    x=group["bin_center"],
                    y=group["pooled_count"],
                    width=group["bin_width"] * 0.92,
                    name=str(label),
                    opacity=0.62,
                    customdata=group[
                        [
                            "bin_left",
                            "bin_right",
                            "pooled_fraction",
                            "pooled_density",
                            "recording_count",
                        ]
                    ],
                    hovertemplate=(
                        "Epoch=%{fullData.name}<br>"
                        "Bin=%{customdata[0]:.3g} to %{customdata[1]:.3g}<br>"
                        "Count=%{y}<br>"
                        "Fraction=%{customdata[2]:.3g}<br>"
                        "Density=%{customdata[3]:.3g}<br>"
                        "Recordings=%{customdata[4]}"
                        "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Count",
            barmode="overlay",
            bargap=0.05,
            margin=dict(l=50, r=30, t=55, b=50),
        )
        return fig

    def build_bar_figure(rows, *, title, yaxis_title):
        frame = pd.DataFrame(rows)
        if frame.empty:
            return None
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=frame["window_label"],
                y=frame["value"],
                customdata=frame[["recording_count", "mean", "median", "sem"]],
                marker_color="#2563eb",
                hovertemplate=(
                    "Epoch=%{x}<br>"
                    "Value=%{y:.3g}<br>"
                    "Recordings=%{customdata[0]}<br>"
                    "Mean=%{customdata[1]:.3g}<br>"
                    "Median=%{customdata[2]:.3g}<br>"
                    "SEM=%{customdata[3]:.3g}"
                    "<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title=yaxis_title,
            margin=dict(l=50, r=30, t=55, b=50),
        )
        return fig

    return (
        Path,
        CHASER_SUMMARY_TABLE,
        SPATIAL_TABLE,
        build_context,
        build_health_report,
        discover_export_catalog,
        mo,
        os,
        pd,
        query_epoch_bout_histogram,
        query_epoch_inter_bout_interval_histogram,
        query_epoch_speed_summary,
        query_export_summary,
        query_group_statistics,
        query_options,
        query_recordings,
        select_export_run_id,
        build_bar_figure,
        build_histogram_figure,
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
    stats_run_id_raw = cli_args.get("stats-run-id", "auto")
    stats_run_id = None if stats_run_id_raw in (None, "", "none", "None") else str(stats_run_id_raw)

    export_catalog = discover_export_catalog(export_root)
    catalog_diagnostics = pd.DataFrame(
        [diagnostic.to_dict() for diagnostic in export_catalog.diagnostics]
    )
    if not export_catalog.entries:
        details = (
            mo.ui.table(catalog_diagnostics, selection=None, page_size=10)
            if not catalog_diagnostics.empty
            else mo.md("No manifest diagnostics were produced.")
        )
        mo.stop(
            True,
            mo.vstack(
                [
                    mo.md("# Palette Group Analytics"),
                    mo.callout(
                        mo.md(
                            f"No selectable analytics exports were found under `{export_catalog.export_root}`. "
                            "The app expects export manifests in `v1/manifests`."
                        ),
                        kind="warn",
                    ),
                    details,
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
    catalog_rows = pd.DataFrame([entry.to_dict() for entry in export_catalog.entries])
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
                    catalog_rows,
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
    CHASER_SUMMARY_TABLE,
    SPATIAL_TABLE,
    build_health_report,
    context,
    export_catalog,
    mo,
    query_export_summary,
    selected_export_run_id,
):
    health = build_health_report(context)
    summary = query_export_summary(context)
    selected_entry = export_catalog.entry(selected_export_run_id)
    collection = summary.get("collection") or {}
    source_recording_count = summary.get("source_recording_count")
    available_tables = set(selected_entry.table_names if selected_entry else ())
    supports_current_panels = {
        SPATIAL_TABLE,
        CHASER_SUMMARY_TABLE,
    }.issubset(available_tables)
    overview = mo.vstack(
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
                    mo.stat(label="Tables", value=f"{len(available_tables):,}"),
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
        ]
    )
    overview
    mo.stop(
        selected_entry is not None and not selected_entry.ready,
        mo.callout(
            mo.md(
                "This export has missing Parquet parts. Its manifest remains visible for "
                "diagnosis, but analysis panels are disabled so they cannot present a partial "
                "dataset as complete."
            ),
            kind="warn",
        ),
    )
    mo.stop(
        not supports_current_panels,
        mo.callout(
            mo.md(
                "This export is discoverable and its provenance is available, but it does not "
                "contain the chaser tables required by the currently implemented panels. "
                "A provider-specific panel can be added without changing dataset discovery."
            ),
            kind="info",
        ),
    )
    analysis_context = context
    return analysis_context, health, summary


@app.cell
def _(analysis_context, mo, query_options):
    options = query_options(analysis_context)
    window_labels = ["All epochs"] + [item["window_label"] for item in options.get("windows", [])]
    bout_metric_items = options.get("epoch_bout_histogram_metrics", [])
    bout_metric_labels = {
        f"{item['label']} [{item['metric']}]": item["metric"]
        for item in bout_metric_items
    }
    default_bout_label = next(iter(bout_metric_labels), "")
    window_picker = mo.ui.dropdown(
        options=window_labels,
        value=window_labels[0],
        label="Epoch",
    )
    bout_metric_picker = mo.ui.dropdown(
        options=list(bout_metric_labels),
        value=default_bout_label,
        label="Bout distribution metric",
    )
    mean_metric_labels = {
        "Bout rate (/min)": "bout_rate_per_min",
        "Mean bout duration (s)": "mean_bout_duration_s",
        "Mean bout distance (mm)": "mean_bout_path_length_mm",
        "Mean inter-bout interval (s)": "mean_inter_bout_interval_s",
        "Mean net heading change (deg)": "mean_bout_net_heading_change_deg",
        "Mean abs heading change (deg)": "mean_abs_bout_net_heading_change_deg",
        "Mean heading path (deg)": "mean_bout_heading_path_deg",
    }
    mean_metric_picker = mo.ui.dropdown(
        options=list(mean_metric_labels),
        value="Bout rate (/min)",
        label="Epoch summary metric",
    )
    stat_labels = {"Mean": "mean", "Median": "median"}
    stat_picker = mo.ui.dropdown(
        options=list(stat_labels),
        value="Mean",
        label="Summary statistic",
    )
    mo.hstack([window_picker, bout_metric_picker, mean_metric_picker, stat_picker])
    return (
        bout_metric_labels,
        bout_metric_picker,
        mean_metric_labels,
        mean_metric_picker,
        options,
        stat_labels,
        stat_picker,
        window_picker,
    )


@app.cell
def _(analysis_context, bout_metric_labels, bout_metric_picker, query_epoch_bout_histogram, window_picker):
    selected_window = None if window_picker.value == "All epochs" else str(window_picker.value)
    selected_bout_metric = bout_metric_labels.get(bout_metric_picker.value, "bout_path_length_mm")
    bout_histogram = query_epoch_bout_histogram(
        analysis_context,
        metric=selected_bout_metric,
        window_label=selected_window,
    )
    return bout_histogram, selected_bout_metric, selected_window


@app.cell
def _(
    build_histogram_figure,
    bout_histogram,
    mo,
    pd,
):
    if not bout_histogram.get("available"):
        _output = mo.md(f"## Bout Metric Distribution\n\n{bout_histogram.get('message', 'No rows.')}")
    else:
        bout_fig = build_histogram_figure(
            bout_histogram["rows"],
            title=f"Bout Metric Distribution: {bout_histogram['metric_label']}",
            xaxis_title=bout_histogram["metric_label"],
        )
        bout_table = pd.DataFrame(bout_histogram["rows"])
        _output = mo.vstack(
            [
                mo.md("## Bout Metric Distribution"),
                bout_fig,
                mo.ui.table(
                    bout_table[bout_table["pooled_count"] > 0],
                    selection=None,
                    page_size=12,
                ),
            ]
        )
    _output
    return


@app.cell
def _(analysis_context, query_epoch_inter_bout_interval_histogram, selected_window):
    ibi_histogram = query_epoch_inter_bout_interval_histogram(
        analysis_context,
        window_label=selected_window,
    )
    return (ibi_histogram,)


@app.cell
def _(build_histogram_figure, ibi_histogram, mo, pd):
    if not ibi_histogram.get("available"):
        _output = mo.md(f"## Inter-Bout Interval Distribution\n\n{ibi_histogram.get('message', 'No rows.')}")
    else:
        ibi_fig = build_histogram_figure(
            ibi_histogram["rows"],
            title="Inter-Bout Interval Distribution",
            xaxis_title="Interval (s)",
        )
        ibi_table = pd.DataFrame(ibi_histogram["rows"])
        _output = mo.vstack(
            [
                mo.md("## Inter-Bout Interval Distribution"),
                ibi_fig,
                mo.ui.table(
                    ibi_table[ibi_table["pooled_count"] > 0],
                    selection=None,
                    page_size=12,
                ),
            ]
        )
    _output
    return


@app.cell
def _(analysis_context, mean_metric_labels, mean_metric_picker, query_epoch_speed_summary, stat_labels, stat_picker):
    epoch_summary_metric = query_epoch_speed_summary(
        analysis_context,
        metric=str(mean_metric_labels.get(mean_metric_picker.value, "bout_rate_per_min")),
        stat=str(stat_labels.get(stat_picker.value, "mean")),
    )
    return (epoch_summary_metric,)


@app.cell
def _(build_bar_figure, epoch_summary_metric, mo, pd):
    if not epoch_summary_metric.get("available"):
        _output = mo.md(f"## Epoch Summary\n\n{epoch_summary_metric.get('message', 'No rows.')}")
    else:
        metric_fig = build_bar_figure(
            epoch_summary_metric["rows"],
            title=f"Epoch Summary: {epoch_summary_metric['metric_label']}",
            yaxis_title=epoch_summary_metric["metric_label"],
        )
        _output = mo.vstack(
            [
                mo.md("## Epoch Summary"),
                metric_fig,
                mo.ui.table(pd.DataFrame(epoch_summary_metric["rows"]), selection=None, page_size=12),
            ]
        )
    _output
    return


@app.cell
def _(analysis_context, mo, pd, query_group_statistics):
    stats = query_group_statistics(analysis_context, metric_family="epoch_behavior")
    if not stats.get("available"):
        _output = mo.md(f"## Epoch Behavior Statistics\n\n{stats.get('message', 'No statistics export found.')}")
    else:
        stats_rows = pd.DataFrame(stats["rows"])
        display_columns = [
            column
            for column in (
                "metric_name",
                "condition_a",
                "condition_b",
                "group",
                "paired_unit_count",
                "mean_difference",
                "median_difference",
                "p_value",
                "rank_biserial",
                "status",
            )
            if column in stats_rows.columns
        ]
        _output = mo.vstack(
            [
                mo.md("## Epoch Behavior Statistics"),
                mo.ui.table(
                    stats_rows[display_columns] if display_columns else stats_rows,
                    selection=None,
                    page_size=12,
                ),
            ]
        )
    _output
    return


@app.cell
def _(analysis_context, mo, pd, query_recordings, summary):
    recordings = query_recordings(analysis_context)
    recording_rows = pd.DataFrame(recordings.get("rows", []))
    table_counts = pd.DataFrame(summary.get("tables", []))
    mo.accordion(
        {
            "Recordings": mo.ui.table(recording_rows, selection=None, page_size=12),
            "Export tables": mo.ui.table(table_counts, selection=None, page_size=16),
        }
    )
    return


if __name__ == "__main__":
    app.run()
