#!/usr/bin/env python3
"""Read-only explorer for one exact validated-behavior statistics generation.

Run with:

    scripts/py -m marimo run \
      apps/marimo/validated_behavior_group_statistics_explorer.py -- \
      --statistics-dir /path/to/exact/grouped-statistics-generation
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

    from apps.marimo.components.validated_behavior_group_statistics import (
        statistics_contrast_rows,
        statistics_dimension_options,
        statistics_metric_options,
        statistics_provenance_rows,
        validated_behavior_statistics_figure,
    )
    from fisheye.group_statistics.validated_behavior_views import (
        ValidatedBehaviorStatisticsViewSource,
        available_statistics_views,
        build_statistics_view_payload,
    )

    return (
        Path,
        ValidatedBehaviorStatisticsViewSource,
        available_statistics_views,
        build_statistics_view_payload,
        mo,
        os,
        pl,
        statistics_contrast_rows,
        statistics_dimension_options,
        statistics_metric_options,
        statistics_provenance_rows,
        validated_behavior_statistics_figure,
    )


@app.cell
def _(Path, ValidatedBehaviorStatisticsViewSource, available_statistics_views, mo, os):
    cli_args = mo.cli_args()
    statistics_dir_raw = cli_args.get(
        "statistics-dir",
        os.environ.get("PALETTE_VALIDATED_BEHAVIOR_STATISTICS_DIR"),
    )
    if not statistics_dir_raw:
        raise ValueError(
            "Required CLI arg is missing. Run with: scripts/py -m marimo run "
            "apps/marimo/validated_behavior_group_statistics_explorer.py -- "
            "--statistics-dir <exact-generation>"
        )
    statistics_dir = Path(str(statistics_dir_raw)).expanduser().resolve()
    source = ValidatedBehaviorStatisticsViewSource.open(statistics_dir)
    view_definitions = available_statistics_views(source)
    if not view_definitions:
        raise ValueError(
            "The exact grouped-statistics generation has no supported views"
        )
    return source, statistics_dir, view_definitions


@app.cell
def _(mo, source, statistics_dir, view_definitions):
    view_label_to_id = {
        f"{definition.label} — {definition.description}": definition.view_id
        for definition in view_definitions
    }
    requested_view = str(mo.cli_args().get("view", view_definitions[0].view_id))
    initial_view_label = next(
        (
            label
            for label, view_id in view_label_to_id.items()
            if view_id == requested_view
        ),
        next(iter(view_label_to_id)),
    )
    view_picker = mo.ui.dropdown(
        options=list(view_label_to_id),
        value=initial_view_label,
        label="Cohort view",
        searchable=True,
    )
    mo.vstack(
        [
            mo.md("# Validated Behavior Cohort Statistics"),
            mo.callout(
                mo.md(
                    "This explorer is **read-only and exploratory**. It consumes one "
                    "exact, digest-validated statistics generation; it does not recompute "
                    "statistics, discover `latest`, update selectors, or mutate sources."
                ),
                kind="info",
            ),
            mo.md(
                f"Statistics: `{source.statistics_run_id}`  \n"
                f"Manifest: `{source.cache_identity}`  \n"
                f"Path: `{statistics_dir}`"
            ),
            view_picker,
        ]
    )
    return view_label_to_id, view_picker


@app.cell
def _(
    build_statistics_view_payload,
    statistics_dimension_options,
    statistics_metric_options,
    source,
    view_label_to_id,
    view_picker,
):
    selected_view_id = view_label_to_id[view_picker.value]
    payload = build_statistics_view_payload(source, selected_view_id)
    metric_label_to_id = statistics_metric_options(payload)
    provider_values = (
        ()
        if selected_view_id == "distance_traveled"
        else statistics_dimension_options(payload, "provider_role")
    )
    behavior_values = statistics_dimension_options(payload, "behavior_role")
    condition_values = statistics_dimension_options(payload, "condition")
    return (
        behavior_values,
        condition_values,
        metric_label_to_id,
        payload,
        provider_values,
        selected_view_id,
    )


@app.cell
def _(
    behavior_values,
    condition_values,
    metric_label_to_id,
    mo,
    payload,
    provider_values,
    selected_view_id,
):
    default_metric_id = str(payload["default_metric_id"])
    default_metric_label = next(
        label
        for label, metric_id in metric_label_to_id.items()
        if metric_id == default_metric_id
    )
    metric_picker = mo.ui.dropdown(
        options=list(metric_label_to_id),
        value=default_metric_label,
        label="Metric",
        searchable=True,
    )

    provider_label_to_value = {
        **({"All providers": None} if selected_view_id != "spatial_occupancy" else {}),
        **{value.title(): value for value in provider_values},
    }
    provider_picker = mo.ui.dropdown(
        options=list(provider_label_to_value),
        value=next(iter(provider_label_to_value)) if provider_label_to_value else None,
        label="Position provider",
    )
    behavior_label_to_value = {
        "All behavior roles": None,
        **{value.title(): value for value in behavior_values},
    }
    behavior_picker = mo.ui.dropdown(
        options=list(behavior_label_to_value),
        value="All behavior roles",
        label="Behavior role",
    )
    condition_label_to_value = {
        str(payload["condition_labels"].get(value, value)): value
        for value in condition_values
    }
    condition_picker = mo.ui.dropdown(
        options=list(condition_label_to_value),
        value=(
            next(iter(condition_label_to_value)) if condition_label_to_value else None
        ),
        label="Epoch",
    )
    occupancy_statistic_picker = mo.ui.dropdown(
        options=["mean", "median"],
        value="mean",
        label="Cohort occupancy statistic",
    )

    controls = [metric_picker]
    if provider_values:
        controls.append(provider_picker)
    if behavior_values:
        controls.append(behavior_picker)
    if selected_view_id == "spatial_occupancy":
        controls.extend([condition_picker, occupancy_statistic_picker])
    mo.hstack(controls, justify="start", align="end", wrap=True)
    return (
        behavior_label_to_value,
        behavior_picker,
        condition_label_to_value,
        condition_picker,
        metric_picker,
        occupancy_statistic_picker,
        provider_label_to_value,
        provider_picker,
    )


@app.cell
def _(
    behavior_label_to_value,
    behavior_picker,
    condition_label_to_value,
    condition_picker,
    metric_label_to_id,
    metric_picker,
    occupancy_statistic_picker,
    payload,
    provider_label_to_value,
    provider_picker,
    selected_view_id,
    validated_behavior_statistics_figure,
):
    selected_metric_id = metric_label_to_id[metric_picker.value]
    selected_provider = (
        provider_label_to_value.get(provider_picker.value)
        if provider_label_to_value
        else None
    )
    selected_behavior_role = behavior_label_to_value.get(behavior_picker.value)
    selected_condition = (
        condition_label_to_value.get(condition_picker.value)
        if selected_view_id == "spatial_occupancy"
        else None
    )
    figure = validated_behavior_statistics_figure(
        payload,
        metric_id=selected_metric_id,
        provider_role=selected_provider,
        behavior_role=selected_behavior_role,
        condition=selected_condition,
        occupancy_statistic=occupancy_statistic_picker.value,
    )
    return (
        figure,
        selected_behavior_role,
        selected_condition,
        selected_metric_id,
        selected_provider,
    )


@app.cell
def _(
    figure,
    mo,
    payload,
    pl,
    selected_behavior_role,
    selected_condition,
    selected_metric_id,
    selected_provider,
    statistics_contrast_rows,
    statistics_provenance_rows,
):
    summary_rows = [
        dict(row)
        for row in payload["descriptive_rows"]
        if row["metric_id"] == selected_metric_id
        and (
            selected_provider is None
            or str(row.get("provider_role", "all")) == selected_provider
        )
        and (
            selected_behavior_role is None
            or str(row.get("behavior_role", "all")) == selected_behavior_role
        )
        and (
            selected_condition is None
            or str(row.get("condition")) == selected_condition
        )
    ]
    summary_preview = summary_rows[:500]
    contrast_rows = statistics_contrast_rows(payload, selected_metric_id)
    provenance_rows = statistics_provenance_rows(payload)
    preview_note = (
        f"Showing the first 500 of {len(summary_rows):,} exact descriptive rows."
        if len(summary_rows) > 500
        else f"Showing all {len(summary_rows):,} exact descriptive rows."
    )
    mo.vstack(
        [
            figure,
            mo.callout(
                mo.md(
                    "Medians and interquartile ranges use equal weight per finite "
                    "`recording_id`. No acquisition-batch adjustment was performed. "
                    "Role colors encode semantic roles, never raw dot colors; provider "
                    "identity is explicit and uses line style where applicable."
                ),
                kind="warn",
            ),
            mo.accordion(
                {
                    f"Descriptive rows — {preview_note}": (
                        mo.ui.table(
                            pl.from_dicts(summary_preview, infer_schema_length=None),
                            selection=None,
                            page_size=15,
                        )
                        if summary_preview
                        else mo.md("No descriptive rows match this selection.")
                    ),
                    "Paired contrasts": (
                        mo.ui.table(
                            pl.from_dicts(contrast_rows, infer_schema_length=None),
                            selection=None,
                            page_size=15,
                        )
                        if contrast_rows
                        else mo.md(
                            "No paired contrasts are defined for this metric family."
                        )
                    ),
                    "Exact provenance": mo.ui.table(
                        pl.from_dicts(provenance_rows, infer_schema_length=None),
                        selection=None,
                        page_size=10,
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
