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
    from apps.marimo.components.training_response import (
        TRAINING_RESPONSE_CATEGORY_FIELDS,
        TRAINING_RESPONSE_METRICS,
        filter_training_response_rows,
        training_response_scatter_figure,
    )
    from fisheye.baseline_strategy.qc import (
        discover_strategy_catalog,
        load_strategy_manifest,
        scan_recording_baseline_samples,
        scan_strategy_qc_rows,
        select_strategy_run_id,
        source_export_context,
    )
    from fisheye.training_response.query import (
        discover_training_response_catalog,
        load_training_response_manifest,
        scan_training_response_qc_rows,
        select_training_response_run_id,
    )

    return (
        Path,
        STRATEGY_CATEGORY_FIELDS,
        STRATEGY_FEATURE_METRICS,
        TRAINING_RESPONSE_CATEGORY_FIELDS,
        TRAINING_RESPONSE_METRICS,
        category_count_figure,
        discover_strategy_catalog,
        discover_training_response_catalog,
        feature_distribution_figure,
        feature_scatter_figure,
        filter_qc_rows,
        filter_training_response_rows,
        load_strategy_manifest,
        load_training_response_manifest,
        mo,
        os,
        pd,
        scan_recording_baseline_samples,
        scan_strategy_qc_rows,
        scan_training_response_qc_rows,
        select_strategy_run_id,
        select_training_response_run_id,
        source_export_context,
        speed_trace_figure,
        trajectory_figure,
        training_response_scatter_figure,
    )


@app.cell
def _(
    Path,
    discover_strategy_catalog,
    discover_training_response_catalog,
    mo,
    os,
    pd,
    select_strategy_run_id,
):
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
    training_response_root = Path(
        str(
            cli_args.get(
                "training-response-root",
                os.environ.get(
                    "PALETTE_TRAINING_RESPONSE_ANALYTICS_ROOT",
                    "/groups/johnson/johnsonlab/palette_training_response_analytics",
                ),
            )
        )
    )
    requested_run_id = str(cli_args.get("analysis-run-id", "latest"))
    requested_training_response_run_id = str(
        cli_args.get("training-response-run-id", "latest")
    )
    catalog = discover_strategy_catalog(strategy_root)
    training_response_catalog = discover_training_response_catalog(
        training_response_root
    )
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
    return (
        export_root,
        label_to_run_id,
        requested_training_response_run_id,
        run_picker,
        strategy_root,
        training_response_catalog,
        training_response_root,
    )


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


@app.cell
def _(
    mo,
    pd,
    requested_training_response_run_id,
    select_training_response_run_id,
    source_context,
    strategy_manifest,
    training_response_catalog,
    training_response_root,
):
    selected_source_export_sha256 = str(
        strategy_manifest.get("source_export_manifest_sha256") or ""
    ).strip()
    selected_source_collection_sha256 = str(
        strategy_manifest.get("source_collection_manifest_sha256") or ""
    ).strip()
    matching_training_entries = [
        entry
        for entry in training_response_catalog.entries
        if entry.ready and entry.source_export_run_id == source_context.export_run_id
        and (
            not selected_source_export_sha256
            or entry.source_export_manifest_sha256 == selected_source_export_sha256
        )
        and (
            not selected_source_collection_sha256
            or entry.source_collection_manifest_sha256
            == selected_source_collection_sha256
        )
    ]
    training_response_warning = None
    training_response_picker = None
    training_response_label_to_run_id = {}
    if matching_training_entries:
        try:
            initial_training_response_run_id = select_training_response_run_id(
                training_response_catalog,
                requested_training_response_run_id,
                source_export_run_id=source_context.export_run_id,
                source_export_manifest_sha256=(
                    selected_source_export_sha256 or None
                ),
                source_collection_manifest_sha256=(
                    selected_source_collection_sha256 or None
                ),
            )
        except ValueError as exc:
            initial_training_response_run_id = select_training_response_run_id(
                training_response_catalog,
                "latest",
                source_export_run_id=source_context.export_run_id,
                source_export_manifest_sha256=(
                    selected_source_export_sha256 or None
                ),
                source_collection_manifest_sha256=(
                    selected_source_collection_sha256 or None
                ),
            )
            training_response_warning = str(exc)
        training_response_label_to_run_id = {
            entry.label: entry.analysis_run_id for entry in matching_training_entries
        }
        initial_training_response_label = next(
            label
            for label, run_id in training_response_label_to_run_id.items()
            if run_id == initial_training_response_run_id
        )
        training_response_picker = mo.ui.dropdown(
            options=list(training_response_label_to_run_id),
            value=initial_training_response_label,
            label="Training-response analysis run",
            searchable=True,
        )

    training_response_diagnostics = [
        {
            "manifest_path": item.manifest_path,
            "code": item.code,
            "message": item.message,
        }
        for item in training_response_catalog.diagnostics
    ]
    training_selector = [
        mo.md("# Whole-training Response QC"),
        mo.md(
            "Read-only, descriptive pre-to-training response profiles. "
            "Closer/farther labels are relative to this export cohort; they do not "
            "establish avoidance, fear, anxiety, or escape success."
        ),
        mo.md(
            f"Training-response root: `{training_response_root.resolve()}`  \n"
            f"Required source export: `{source_context.export_run_id}`"
        ),
    ]
    if training_response_picker is not None:
        training_selector.append(training_response_picker)
    else:
        training_selector.append(
            mo.callout(
                "No ready training-response run matches the selected baseline run's "
                "source export. Baseline QC remains available above.",
                kind="warn",
            )
        )
    if training_response_warning:
        training_selector.append(mo.callout(training_response_warning, kind="warn"))
    training_selector.append(
        mo.accordion(
            {
                "Matching immutable runs": (
                    mo.ui.table(
                        pd.DataFrame(
                            [
                                {
                                    "analysis_run_id": entry.analysis_run_id,
                                    "source_export_run_id": entry.source_export_run_id,
                                    "source_export_manifest_sha256": (
                                        entry.source_export_manifest_sha256
                                    ),
                                    "created_at_utc": entry.created_at_utc,
                                    "row_count": entry.row_count,
                                }
                                for entry in matching_training_entries
                            ]
                        ),
                        selection=None,
                        page_size=10,
                    )
                    if matching_training_entries
                    else mo.md("No matching runs.")
                ),
                "Rejected manifests": (
                    mo.ui.table(
                        pd.DataFrame(training_response_diagnostics),
                        selection=None,
                        page_size=10,
                    )
                    if training_response_diagnostics
                    else mo.md("No manifests were rejected.")
                ),
            }
        )
    )
    mo.vstack(training_selector)
    return training_response_label_to_run_id, training_response_picker


@app.cell
def _(
    load_training_response_manifest,
    mo,
    scan_training_response_qc_rows,
    training_response_label_to_run_id,
    training_response_picker,
    training_response_root,
):
    mo.stop(training_response_picker is None)
    selected_training_response_run_id = training_response_label_to_run_id[
        training_response_picker.value
    ]
    training_response_manifest = load_training_response_manifest(
        training_response_root, selected_training_response_run_id
    )
    training_response_rows = (
        scan_training_response_qc_rows(
            training_response_root, selected_training_response_run_id
        )
        .sort(["protocol_name", "recording_id"])
        .collect()
        .to_dicts()
    )
    return (
        selected_training_response_run_id,
        training_response_manifest,
        training_response_rows,
    )


@app.cell
def _(
    TRAINING_RESPONSE_CATEGORY_FIELDS,
    TRAINING_RESPONSE_METRICS,
    mo,
    selected_training_response_run_id,
    training_response_manifest,
    training_response_rows,
):
    training_protocol_values = sorted(
        {str(row.get("protocol_name") or "unknown") for row in training_response_rows}
    )
    training_status_values = sorted(
        {
            str(row.get("classification_status") or "unknown")
            for row in training_response_rows
        }
    )
    training_protocol_picker = mo.ui.multiselect(
        options=training_protocol_values,
        value=training_protocol_values,
        label="Protocols",
    )
    training_status_picker = mo.ui.multiselect(
        options=training_status_values,
        value=training_status_values,
        label="Tracking/classification status",
    )
    training_category_labels = {
        label: field for field, label in TRAINING_RESPONSE_CATEGORY_FIELDS.items()
    }
    training_metric_labels = {
        label: field for field, label in TRAINING_RESPONSE_METRICS.items()
    }
    training_category_picker = mo.ui.dropdown(
        options=list(training_category_labels),
        value=TRAINING_RESPONSE_CATEGORY_FIELDS["primary_training_profile"],
        label="Response category",
    )
    training_metric_picker = mo.ui.dropdown(
        options=list(training_metric_labels),
        value=TRAINING_RESPONSE_METRICS["mean_speed_mm_s_log2_ratio"],
        label="Continuous feature",
    )
    training_feature_config = training_response_manifest.get("feature_config") or {}
    training_complete_count = sum(
        row.get("classification_status") == "complete"
        for row in training_response_rows
    )
    training_invalid_count = sum(
        row.get("classification_status") == "invalid"
        for row in training_response_rows
    )
    mo.vstack(
        [
            mo.md(f"## Training cohort overview · `{selected_training_response_run_id}`"),
            mo.hstack(
                [
                    mo.stat(label="Recordings", value=f"{len(training_response_rows):,}"),
                    mo.stat(
                        label="Classification complete",
                        value=f"{training_complete_count:,}",
                    ),
                    mo.stat(label="Tracking invalid", value=f"{training_invalid_count:,}"),
                    mo.stat(
                        label="Minimum valid coverage",
                        value=(
                            f"{float(training_feature_config.get('min_valid_position_fraction', 0)):.0%}"
                        ),
                    ),
                ]
            ),
            mo.hstack(
                [
                    training_protocol_picker,
                    training_status_picker,
                    training_category_picker,
                    training_metric_picker,
                ],
                widths="equal",
            ),
            mo.callout(
                "Temporal adaptation and habituation are unavailable in this run because "
                "training-period time bins or samples were not exported.",
                kind="info",
            ),
        ]
    )
    return (
        training_category_labels,
        training_category_picker,
        training_metric_labels,
        training_metric_picker,
        training_protocol_picker,
        training_status_picker,
    )


@app.cell
def _(
    category_count_figure,
    feature_distribution_figure,
    filter_training_response_rows,
    mo,
    pd,
    training_category_labels,
    training_category_picker,
    training_metric_labels,
    training_metric_picker,
    training_protocol_picker,
    training_response_rows,
    training_response_scatter_figure,
    training_status_picker,
):
    filtered_training_response_rows = filter_training_response_rows(
        training_response_rows,
        protocols=training_protocol_picker.value,
        statuses=training_status_picker.value,
    )
    selected_training_category_label = training_category_picker.value
    selected_training_category_key = training_category_labels[
        selected_training_category_label
    ]
    selected_training_metric_label = training_metric_picker.value
    selected_training_metric_key = training_metric_labels[selected_training_metric_label]
    training_category_figure = category_count_figure(
        filtered_training_response_rows,
        category_key=selected_training_category_key,
        title=f"{selected_training_category_label} by protocol",
    )
    training_distribution_figure = feature_distribution_figure(
        filtered_training_response_rows,
        metric=selected_training_metric_key,
        label=selected_training_metric_label,
    )
    training_scatter_figure = training_response_scatter_figure(
        filtered_training_response_rows
    )
    invalid_training_rows = [
        {
            "recording_id": row.get("recording_id"),
            "protocol_name": row.get("protocol_name"),
            "pre_tracking_dropout_fraction": row.get(
                "pre_tracking_dropout_fraction"
            ),
            "training_tracking_dropout_fraction": row.get(
                "training_tracking_dropout_fraction"
            ),
            "reason": row.get("classification_reason"),
        }
        for row in training_response_rows
        if row.get("classification_status") != "complete"
    ]
    no_training_rows = mo.callout(
        "No finite rows match the current filters.", kind="warn"
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    (
                        mo.ui.plotly(training_category_figure)
                        if training_category_figure
                        else no_training_rows
                    ),
                    (
                        mo.ui.plotly(training_distribution_figure)
                        if training_distribution_figure
                        else no_training_rows
                    ),
                ],
                widths="equal",
            ),
            (
                mo.ui.plotly(training_scatter_figure)
                if training_scatter_figure
                else no_training_rows
            ),
            mo.accordion(
                {
                    "Tracking-invalid recordings": mo.ui.table(
                        pd.DataFrame(invalid_training_rows),
                        selection=None,
                        page_size=12,
                    ),
                    "Filtered training-response features": mo.ui.table(
                        pd.DataFrame(filtered_training_response_rows),
                        selection=None,
                        page_size=12,
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
