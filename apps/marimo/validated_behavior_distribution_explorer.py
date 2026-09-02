#!/usr/bin/env python3
"""Read-only explorer for one exact behavior-distribution generation.

Run with:

    scripts/py -m marimo run \
      apps/marimo/validated_behavior_distribution_explorer.py -- \
      --distribution-dir /path/to/exact/distribution-generation
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from apps.marimo.components.validated_behavior_distributions import (
        distribution_dimension_options,
        distribution_metric_options,
        distribution_provenance_rows,
        validated_behavior_distribution_figure,
        validated_behavior_motion_trace_figure,
    )
    from fisheye.group_statistics.validated_behavior_distribution_views import (
        COHORT_STATISTIC_LABELS,
        DEFAULT_DISPLAY_RANGE,
        DISPLAY_RANGE_LABELS,
        ValidatedBehaviorDistributionViewSource,
        available_distribution_metrics,
        build_distribution_view_payload,
        build_motion_trace_payload,
        distribution_recording_ids,
    )

    return (
        COHORT_STATISTIC_LABELS,
        DEFAULT_DISPLAY_RANGE,
        DISPLAY_RANGE_LABELS,
        Path,
        ValidatedBehaviorDistributionViewSource,
        available_distribution_metrics,
        build_distribution_view_payload,
        build_motion_trace_payload,
        distribution_dimension_options,
        distribution_metric_options,
        distribution_provenance_rows,
        distribution_recording_ids,
        mo,
        os,
        pl,
        validated_behavior_distribution_figure,
        validated_behavior_motion_trace_figure,
    )


@app.cell
def _(Path, ValidatedBehaviorDistributionViewSource, mo, os):
    cli_args = mo.cli_args()
    distribution_dir_raw = cli_args.get(
        "distribution-dir",
        os.environ.get("PALETTE_VALIDATED_BEHAVIOR_DISTRIBUTION_DIR"),
    )
    if not distribution_dir_raw:
        raise ValueError(
            "Required CLI arg is missing. Run with: scripts/py -m marimo run "
            "apps/marimo/validated_behavior_distribution_explorer.py -- "
            "--distribution-dir <exact-generation>"
        )
    distribution_dir = Path(str(distribution_dir_raw)).expanduser().resolve()
    distribution_source = ValidatedBehaviorDistributionViewSource.open(distribution_dir)
    return distribution_dir, distribution_source


@app.cell
def _(
    available_distribution_metrics,
    distribution_dir,
    distribution_metric_options,
    distribution_source,
    mo,
):
    distribution_metrics = available_distribution_metrics(distribution_source)
    metric_label_to_id = distribution_metric_options(distribution_metrics)
    requested_metric = str(
        mo.cli_args().get("metric", distribution_metrics[0]["metric_id"])
    )
    initial_metric_label = next(
        (
            label
            for label, metric_id in metric_label_to_id.items()
            if metric_id == requested_metric
        ),
        next(iter(metric_label_to_id)),
    )
    metric_picker = mo.ui.dropdown(
        options=list(metric_label_to_id),
        value=initial_metric_label,
        label="Metric",
        searchable=True,
    )
    mo.vstack(
        [
            mo.md("# Validated Behavior Distributions"),
            mo.callout(
                mo.md(
                    "This explorer is **read-only and exploratory**. Histograms "
                    "come from one exact digest-validated distribution generation. "
                    "The default is an equal-weight mean across recording-normalized "
                    "fractions; pooled observations are explicitly diagnostic."
                ),
                kind="info",
            ),
            mo.md(
                f"Distribution: `{distribution_source.distribution_run_id}`  \n"
                f"Manifest: `{distribution_source.cache_identity}`  \n"
                f"Path: `{distribution_dir}`"
            ),
            metric_picker,
        ]
    )
    return distribution_metrics, metric_label_to_id, metric_picker


@app.cell
def _(distribution_metrics, metric_label_to_id, metric_picker, mo):
    selected_metric_id = metric_label_to_id[metric_picker.value]
    selected_metric = next(
        metric
        for metric in distribution_metrics
        if metric["metric_id"] == selected_metric_id
    )
    weighting_label_to_id = {
        {"event": "Event weighted", "frame": "Frame weighted", "time": "Time weighted"}[
            str(weighting)
        ]: str(weighting)
        for weighting in selected_metric["weighting_ids"]
    }
    weighting_picker = mo.ui.dropdown(
        options=list(weighting_label_to_id),
        value=next(iter(weighting_label_to_id)),
        label="Observation weighting",
    )
    weighting_picker
    return selected_metric, selected_metric_id, weighting_label_to_id, weighting_picker


@app.cell
def _(
    build_distribution_view_payload,
    distribution_source,
    selected_metric_id,
    weighting_label_to_id,
    weighting_picker,
):
    selected_weighting_id = weighting_label_to_id[weighting_picker.value]
    distribution_payload = build_distribution_view_payload(
        distribution_source, selected_metric_id, selected_weighting_id
    )
    return distribution_payload, selected_weighting_id


@app.cell
def _(
    COHORT_STATISTIC_LABELS,
    DEFAULT_DISPLAY_RANGE,
    DISPLAY_RANGE_LABELS,
    distribution_dimension_options,
    distribution_payload,
    mo,
):
    statistic_label_to_id = {
        label: statistic for statistic, label in COHORT_STATISTIC_LABELS.items()
    }
    statistic_picker = mo.ui.dropdown(
        options=list(statistic_label_to_id),
        value=COHORT_STATISTIC_LABELS["mean_recording_fraction"],
        label="Cohort statistic",
    )
    display_range_label_to_id = {
        label: range_id for range_id, label in DISPLAY_RANGE_LABELS.items()
    }
    display_range_picker = mo.ui.dropdown(
        options=list(display_range_label_to_id),
        value=DISPLAY_RANGE_LABELS[DEFAULT_DISPLAY_RANGE],
        label="X-axis range",
    )
    provider_values = distribution_dimension_options(
        distribution_payload, "provider_role"
    )
    provider_label_to_value = {
        "All providers": None,
        **{value.title(): value for value in provider_values},
    }
    provider_picker = mo.ui.dropdown(
        options=list(provider_label_to_value),
        value="All providers",
        label="Position provider",
    )
    role_values = distribution_dimension_options(distribution_payload, "behavior_role")
    role_label_to_value = {
        "All behavior roles": None,
        **{value.title(): value for value in role_values},
    }
    role_picker = mo.ui.dropdown(
        options=list(role_label_to_value),
        value="All behavior roles",
        label="Chaser role",
    )
    iqr_picker = mo.ui.checkbox(value=True, label="Show recording IQR")
    controls = [statistic_picker, display_range_picker]
    if provider_values:
        controls.append(provider_picker)
    if role_values:
        controls.append(role_picker)
    controls.append(iqr_picker)
    mo.hstack(controls, justify="start", align="end", wrap=True)
    return (
        display_range_label_to_id,
        display_range_picker,
        iqr_picker,
        provider_label_to_value,
        provider_picker,
        role_label_to_value,
        role_picker,
        statistic_label_to_id,
        statistic_picker,
    )


@app.cell
def _(
    display_range_label_to_id,
    display_range_picker,
    distribution_payload,
    iqr_picker,
    provider_label_to_value,
    provider_picker,
    role_label_to_value,
    role_picker,
    statistic_label_to_id,
    statistic_picker,
    validated_behavior_distribution_figure,
):
    selected_statistic = statistic_label_to_id[statistic_picker.value]
    selected_display_range = display_range_label_to_id[display_range_picker.value]
    selected_provider = provider_label_to_value[provider_picker.value]
    selected_role = role_label_to_value[role_picker.value]
    distribution_figure = validated_behavior_distribution_figure(
        distribution_payload,
        cohort_statistic=selected_statistic,
        provider_role=selected_provider,
        behavior_role=selected_role,
        show_recording_iqr=iqr_picker.value,
        display_range_id=selected_display_range,
    )
    selected_effective_display_range = distribution_figure.layout.meta["display_range"][
        "effective_display_range_id"
    ]
    return (
        distribution_figure,
        selected_effective_display_range,
        selected_provider,
        selected_role,
        selected_statistic,
    )


@app.cell
def _(
    distribution_figure,
    distribution_payload,
    distribution_provenance_rows,
    mo,
    pl,
    selected_effective_display_range,
    selected_statistic,
):
    support_rows = distribution_payload["recording_support_rows"]
    finite_support = [row for row in support_rows if row["support_state"] == "finite"]
    summary = {
        "recording support rows": len(support_rows),
        "finite support rows": len(finite_support),
        "candidate observations": sum(
            int(row["candidate_count"]) for row in support_rows
        ),
        "valid observations": sum(int(row["valid_count"]) for row in support_rows),
        "excluded observations": sum(
            int(row["excluded_count"]) for row in support_rows
        ),
    }
    warning = (
        mo.callout(
            mo.md(
                "**Pooled observations are diagnostic.** Long recordings and groups "
                "with more events receive more weight in this view; this is not the "
                "default experimental-unit summary."
            ),
            kind="warn",
        )
        if selected_statistic == "pooled_fraction"
        else mo.callout(
            mo.md(
                "Each finite `recording_id` contributes equally after its histogram "
                "is normalized. The shaded band is the per-bin recording IQR."
            ),
            kind="info",
        )
    )
    range_notice = (
        mo.callout(
            mo.md(
                "**Central x-view is display-only.** It retains at least 99% of "
                "the equal-recording mass in every displayed series using whole "
                "sealed bins. Choose **Full evidence range** to inspect all tails."
            ),
            kind="info",
        )
        if selected_effective_display_range == "central_99"
        else mo.callout(
            mo.md(
                "The x-axis spans every sealed histogram bin. This also occurs when "
                "the central view is requested for an already logarithmic axis. "
                "Plotly zoom does not change the evidence or cohort statistic."
            ),
            kind="info",
        )
    )
    mo.vstack(
        [
            mo.ui.plotly(distribution_figure),
            range_notice,
            warning,
            mo.accordion(
                {
                    "Support summary": mo.ui.table(
                        pl.from_dicts(
                            [
                                {"quantity": key, "value": value}
                                for key, value in summary.items()
                            ]
                        ),
                        selection=None,
                    ),
                    "Exact provenance": mo.ui.table(
                        pl.from_dicts(
                            distribution_provenance_rows(distribution_payload)
                        ),
                        selection=None,
                        page_size=20,
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(
    distribution_metrics,
    distribution_recording_ids,
    distribution_source,
    mo,
):
    trace_metrics = tuple(
        metric
        for metric in distribution_metrics
        if metric["source_surface"] == "provider_motion_samples"
    )
    trace_metric_label_to_id = {
        str(metric["interpretation"]): str(metric["metric_id"])
        for metric in trace_metrics
    }
    recording_ids = distribution_recording_ids(distribution_source)
    trace_metric_picker = mo.ui.dropdown(
        options=list(trace_metric_label_to_id),
        value=next(iter(trace_metric_label_to_id)),
        label="Trace metric",
        searchable=True,
    )
    recording_picker = mo.ui.dropdown(
        options=list(recording_ids),
        value=recording_ids[0],
        label="Recording",
        searchable=True,
    )
    coordinate_label_to_id = {
        "Acquisition frame": "frame",
        "Session time (s)": "time",
    }
    coordinate_picker = mo.ui.dropdown(
        options=list(coordinate_label_to_id),
        value="Session time (s)",
        label="Trace x-axis",
    )
    trace_points_picker = mo.ui.slider(
        start=1000,
        stop=20000,
        step=1000,
        value=5000,
        label="Maximum display points",
        show_value=True,
    )
    trace_load = mo.ui.run_button(label="Load exact recording trace")
    mo.vstack(
        [
            mo.md("## Frame/time trace"),
            mo.md(
                "This optional view uses the exact same Phase-C rows for either "
                "frame ID or session seconds. The coordinate switch is display-only; "
                "bounded decimation is recorded in the trace payload."
            ),
            mo.hstack(
                [
                    trace_metric_picker,
                    recording_picker,
                    coordinate_picker,
                    trace_points_picker,
                    trace_load,
                ],
                justify="start",
                align="end",
                wrap=True,
            ),
        ]
    )
    return (
        coordinate_label_to_id,
        coordinate_picker,
        recording_picker,
        trace_load,
        trace_metric_label_to_id,
        trace_metric_picker,
        trace_points_picker,
    )


@app.cell
def _(
    build_motion_trace_payload,
    coordinate_label_to_id,
    coordinate_picker,
    distribution_source,
    mo,
    recording_picker,
    trace_load,
    trace_metric_label_to_id,
    trace_metric_picker,
    trace_points_picker,
):
    trace_payload = None
    if trace_load.value:
        trace_payload = build_motion_trace_payload(
            distribution_source,
            metric_id=trace_metric_label_to_id[trace_metric_picker.value],
            recording_id=recording_picker.value,
            coordinate_id=coordinate_label_to_id[coordinate_picker.value],
            max_display_points=int(trace_points_picker.value),
        )
    trace_status = (
        mo.md("Select the trace controls and press **Load exact recording trace**.")
        if trace_payload is None
        else mo.md(
            f"Loaded `{trace_payload['display_point_count']:,}` display points from "
            f"`{trace_payload['source_row_count']:,}` exact rows."
        )
    )
    trace_status
    return trace_payload


@app.cell
def _(mo, trace_payload, validated_behavior_motion_trace_figure):
    if trace_payload is None:
        trace_output = mo.md("")
    else:
        trace_output = mo.ui.plotly(
            validated_behavior_motion_trace_figure(trace_payload)
        )
    trace_output
    return


if __name__ == "__main__":
    app.run()
