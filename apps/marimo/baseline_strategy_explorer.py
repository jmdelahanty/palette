#!/usr/bin/env python3
"""Read-only Marimo QC explorer for immutable baseline-strategy runs.

Run with:

    scripts/py -m marimo run apps/marimo/baseline_strategy_explorer.py -- \
      --strategy-root /groups/johnson/johnsonlab/palette_strategy_analytics \
      --analysis-run-id latest \
      --export-root /groups/johnson/johnsonlab/palette_analytics
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

    from apps.marimo.components.baseline_strategy import (
        STRATEGY_CATEGORY_FIELDS,
        STRATEGY_FEATURE_METRICS,
        category_count_figure,
        feature_distribution_figure,
        feature_scatter_figure,
        filter_qc_rows,
        speed_trace_figure,
        trajectory_figure,
    )
    from fisheye.baseline_strategy.qc import (
        discover_strategy_catalog,
        load_strategy_manifest,
        scan_recording_baseline_samples,
        scan_strategy_qc_rows,
        select_strategy_run_id,
        source_export_context,
    )

    return (
        Path,
        STRATEGY_CATEGORY_FIELDS,
        STRATEGY_FEATURE_METRICS,
        category_count_figure,
        discover_strategy_catalog,
        feature_distribution_figure,
        feature_scatter_figure,
        filter_qc_rows,
        load_strategy_manifest,
        mo,
        os,
        pd,
        scan_recording_baseline_samples,
        scan_strategy_qc_rows,
        select_strategy_run_id,
        source_export_context,
        speed_trace_figure,
        trajectory_figure,
    )


@app.cell
def _(Path, discover_strategy_catalog, mo, os, pd, select_strategy_run_id):
    cli_args = mo.cli_args()
    strategy_root = Path(
        str(
            cli_args.get(
                "strategy-root",
                os.environ.get(
                    "PALETTE_STRATEGY_ANALYTICS_ROOT",
                    "/groups/johnson/johnsonlab/palette_strategy_analytics",
                ),
            )
        )
    )
    export_root = Path(
        str(
            cli_args.get(
                "export-root",
                os.environ.get(
                    "PALETTE_ANALYTICS_EXPORT_ROOT",
                    "/groups/johnson/johnsonlab/palette_analytics",
                ),
            )
        )
    )
    requested_run_id = str(cli_args.get("analysis-run-id", "latest"))
    catalog = discover_strategy_catalog(strategy_root)
    diagnostic_rows = [
        {
            "manifest_path": item.manifest_path,
            "code": item.code,
            "message": item.message,
        }
        for item in catalog.diagnostics
    ]
    if not any(entry.ready for entry in catalog.entries):
        mo.stop(
            True,
            mo.vstack(
                [
                    mo.md("# Baseline Strategy QC"),
                    mo.callout(
                        mo.md(f"No ready analysis runs exist under `{catalog.output_root}`."),
                        kind="warn",
                    ),
                    (
                        mo.ui.table(pd.DataFrame(diagnostic_rows), selection=None)
                        if diagnostic_rows
                        else mo.md("No manifest diagnostics were available.")
                    ),
                ]
            ),
        )

    selection_warning = None
    try:
        initial_run_id = select_strategy_run_id(catalog, requested_run_id)
    except ValueError as exc:
        initial_run_id = select_strategy_run_id(catalog, "latest")
        selection_warning = str(exc)
    label_to_run_id = {
        entry.label: entry.analysis_run_id for entry in catalog.entries if entry.ready
    }
    initial_label = next(
        label for label, run_id in label_to_run_id.items() if run_id == initial_run_id
    )
    run_picker = mo.ui.dropdown(
        options=list(label_to_run_id),
        value=initial_label,
        label="Strategy analysis run",
        searchable=True,
    )
    selector = [
        mo.md("# Baseline Strategy QC"),
        mo.md(
            "Read-only inspection of descriptive pre-stimulus behavior. "
            "These labels do not support anxiety inference."
        ),
        mo.md(f"Strategy root: `{catalog.output_root}`  \nExport root: `{export_root.resolve()}`"),
        run_picker,
    ]
    if selection_warning:
        selector.append(mo.callout(selection_warning, kind="warn"))
    selector.append(
        mo.accordion(
            {
                "Available immutable runs": mo.ui.table(
                    pd.DataFrame(
                        [
                            {
                                "analysis_run_id": entry.analysis_run_id,
                                "source_export_run_id": entry.source_export_run_id,
                                "created_at_utc": entry.created_at_utc,
                                "row_count": entry.row_count,
                                "ready": entry.ready,
                            }
                            for entry in catalog.entries
                        ]
                    ),
                    selection=None,
                    page_size=10,
                ),
                "Rejected manifests": (
                    mo.ui.table(pd.DataFrame(diagnostic_rows), selection=None, page_size=10)
                    if diagnostic_rows
                    else mo.md("No manifests were rejected.")
                ),
            }
        )
    )
    mo.vstack(selector)
    return export_root, label_to_run_id, run_picker, strategy_root


@app.cell
def _(
    export_root,
    label_to_run_id,
    load_strategy_manifest,
    run_picker,
    scan_strategy_qc_rows,
    source_export_context,
    strategy_root,
):
    selected_run_id = label_to_run_id[run_picker.value]
    strategy_manifest = load_strategy_manifest(strategy_root, selected_run_id)
    source_context = source_export_context(
        strategy_root,
        selected_run_id,
        authorized_export_root=export_root,
    )
    qc_rows = (
        scan_strategy_qc_rows(
            strategy_root,
            selected_run_id,
            recording_protocols=source_context.recording_protocols,
        )
        .sort(["protocol_name", "recording_id"])
        .collect()
        .to_dicts()
    )
    return qc_rows, selected_run_id, source_context, strategy_manifest


@app.cell
def _(
    STRATEGY_CATEGORY_FIELDS,
    STRATEGY_FEATURE_METRICS,
    mo,
    qc_rows,
    selected_run_id,
    strategy_manifest,
):
    protocol_values = sorted(
        {str(row.get("protocol_name") or "unknown") for row in qc_rows}
    )
    status_values = sorted(
        {str(row.get("classification_status") or "unknown") for row in qc_rows}
    )
    recording_values = sorted(
        str(row["recording_id"]) for row in qc_rows if row.get("recording_id")
    )
    protocol_picker = mo.ui.multiselect(
        options=protocol_values,
        value=protocol_values,
        label="Protocols",
    )
    status_picker = mo.ui.multiselect(
        options=status_values,
        value=status_values,
        label="Tracking/classification status",
    )
    category_labels = {
        label: field for field, label in STRATEGY_CATEGORY_FIELDS.items()
    }
    metric_labels = {
        label: field for field, label in STRATEGY_FEATURE_METRICS.items()
    }
    category_picker = mo.ui.dropdown(
        options=list(category_labels),
        value=STRATEGY_CATEGORY_FIELDS["primary_strategy"],
        label="Category",
    )
    metric_picker = mo.ui.dropdown(
        options=list(metric_labels),
        value=STRATEGY_FEATURE_METRICS["wall_fraction"],
        label="Continuous feature",
    )
    recording_picker = mo.ui.dropdown(
        options=recording_values,
        value=recording_values[0] if recording_values else None,
        label="Recording drilldown",
        searchable=True,
    )
    feature_config = strategy_manifest.get("feature_config") or {}
    classification_complete_count = sum(
        row.get("classification_status") == "complete" for row in qc_rows
    )
    classification_invalid_count = sum(
        row.get("classification_status") == "invalid" for row in qc_rows
    )
    mo.vstack(
        [
            mo.md(f"## Cohort overview · `{selected_run_id}`"),
            mo.hstack(
                [
                    mo.stat(label="Recordings", value=f"{len(qc_rows):,}"),
                    mo.stat(
                        label="Classification complete",
                        value=f"{classification_complete_count:,}",
                    ),
                    mo.stat(
                        label="Tracking invalid",
                        value=f"{classification_invalid_count:,}",
                    ),
                    mo.stat(
                        label="Minimum valid coverage",
                        value=f"{float(feature_config.get('min_valid_position_fraction', 0)):.0%}",
                    ),
                ]
            ),
            mo.hstack(
                [protocol_picker, status_picker, category_picker, metric_picker],
                widths="equal",
            ),
            recording_picker,
        ]
    )
    return (
        category_labels,
        category_picker,
        metric_labels,
        metric_picker,
        protocol_picker,
        recording_picker,
        status_picker,
    )


@app.cell
def _(
    category_labels,
    category_count_figure,
    category_picker,
    feature_distribution_figure,
    feature_scatter_figure,
    filter_qc_rows,
    metric_labels,
    metric_picker,
    mo,
    pd,
    protocol_picker,
    qc_rows,
    status_picker,
):
    filtered_rows = filter_qc_rows(
        qc_rows,
        protocols=protocol_picker.value,
        statuses=status_picker.value,
    )
    category_label = category_picker.value
    category_key = category_labels[category_label]
    metric_label = metric_picker.value
    metric_key = metric_labels[metric_label]
    category_figure = category_count_figure(
        filtered_rows,
        category_key=category_key,
        title=f"{category_label} by protocol",
    )
    distribution_figure = feature_distribution_figure(
        filtered_rows,
        metric=metric_key,
        label=metric_label,
    )
    scatter_figure = feature_scatter_figure(filtered_rows)
    invalid_rows = [
        {
            "recording_id": row.get("recording_id"),
            "protocol_name": row.get("protocol_name"),
            "tracking_dropout_fraction": row.get("tracking_dropout_fraction"),
            "reason": row.get("classification_reason"),
        }
        for row in qc_rows
        if row.get("classification_status") != "complete"
    ]
    empty_plot = mo.callout("No finite rows match the current filters.", kind="warn")
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.ui.plotly(category_figure) if category_figure else empty_plot,
                    (
                        mo.ui.plotly(distribution_figure)
                        if distribution_figure
                        else empty_plot
                    ),
                ],
                widths="equal",
            ),
            mo.ui.plotly(scatter_figure) if scatter_figure else empty_plot,
            mo.accordion(
                {
                    "Tracking-invalid recordings": mo.ui.table(
                        pd.DataFrame(invalid_rows), selection=None, page_size=12
                    ),
                    "Filtered recording features": mo.ui.table(
                        pd.DataFrame(filtered_rows), selection=None, page_size=12
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(recording_picker, scan_recording_baseline_samples, source_context):
    selected_recording_id = str(recording_picker.value or "")
    selected_samples = (
        scan_recording_baseline_samples(source_context, selected_recording_id)
        .collect()
        .to_dicts()
        if selected_recording_id
        else []
    )
    return selected_recording_id, selected_samples


@app.cell
def _(
    mo,
    pd,
    qc_rows,
    selected_recording_id,
    selected_samples,
    speed_trace_figure,
    trajectory_figure,
):
    selected_metadata = [
        row for row in qc_rows if row.get("recording_id") == selected_recording_id
    ]
    trajectory = trajectory_figure(
        selected_samples, recording_id=selected_recording_id
    )
    speed = speed_trace_figure(selected_samples, recording_id=selected_recording_id)
    mo.vstack(
        [
            mo.md(f"## Recording drilldown · `{selected_recording_id}`"),
            mo.md(
                f"Collected `{len(selected_samples):,}` manifest-declared portable samples "
                "after recording-level predicate pushdown."
            ),
            mo.hstack(
                [
                    (
                        mo.ui.plotly(trajectory)
                        if trajectory
                        else mo.callout("No valid trajectory samples.", kind="warn")
                    ),
                    (
                        mo.ui.plotly(speed)
                        if speed
                        else mo.callout("No valid speed samples.", kind="warn")
                    ),
                ],
                widths="equal",
            ),
            mo.ui.table(pd.DataFrame(selected_metadata), selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
