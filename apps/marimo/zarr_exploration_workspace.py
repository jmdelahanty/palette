#!/usr/bin/env python3
"""Editable, read-only-mounted workspace for an arbitrary source Zarr."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    from apps.marimo.components.common import png_bytes_to_markdown_image
    from apps.marimo.components.goodcopbadcop_chaser import (
        build_chaser_gaze_tracking_output,
        discover_chaser_gaze_tracking_components,
        load_chaser_gaze_tracking_view,
    )
    from apps.marimo.components.zarr_workspace import ZarrExplorationWorkspace

    return (
        Path,
        ZarrExplorationWorkspace,
        build_chaser_gaze_tracking_output,
        discover_chaser_gaze_tracking_components,
        go,
        load_chaser_gaze_tracking_view,
        mo,
        np,
        pl,
        png_bytes_to_markdown_image,
    )


@app.cell(hide_code=True)
def _(Path, ZarrExplorationWorkspace, mo):
    cli_args = mo.cli_args()
    source_raw = cli_args.get("zarr-path")
    if not source_raw:
        raise ValueError(
            "Required --zarr-path is missing. Launch with: "
            "pixi run -e recording zarr-workspace -- --zarr-path <source.zarr>"
        )
    source_path = Path(str(source_raw))
    if not source_path.is_dir():
        raise ValueError(f"Source Zarr directory was not found: {source_path}")
    zarr_workspace = ZarrExplorationWorkspace.open(source_path)
    # This stable handle is defined in a hidden implementation cell so the
    # visible starter cell can be freely replaced by a person or Pair agent.
    exploration = zarr_workspace
    return exploration, source_path, zarr_workspace


@app.cell(hide_code=True)
def _(mo, source_path, zarr_workspace):
    mo.vstack(
        [
            mo.md("# Palette Zarr Exploration Workspace"),
            mo.callout(
                mo.md(
                    f"The selected dataset is mounted read-only at `{source_path}`. "
                    "Notebook edits and derived files may be saved beneath `/workspace`; "
                    "the source Zarr cannot be modified from this session."
                ),
                kind="info",
            ),
            mo.md(
                "Known Palette analysis contracts open in a guided semantic view. "
                "The physical Zarr hierarchy remains available under **Advanced "
                "storage**. Metadata browsing does not load dense frame values, and "
                "all preview and trace reads remain bounded."
            ),
            mo.tree(zarr_workspace.summary(), label="Dataset summary"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(discover_chaser_gaze_tracking_components, mo, source_path, zarr_workspace):
    analysis_dataset_rows = zarr_workspace.analysis_datasets(
        max_runs=100,
        max_tracks_per_run=100,
    )
    eye_angle_run_rows = [
        row
        for row in zarr_workspace.eye_angle_runs(max_runs=100)
        if row.get("frame_angles_path")
    ]
    gaze_tracking_rows = list(
        discover_chaser_gaze_tracking_components(source_path)
    )
    workspace_mode_options = (
        ["Guided analyses", "Advanced storage"]
        if eye_angle_run_rows or analysis_dataset_rows or gaze_tracking_rows
        else ["Advanced storage"]
    )
    workspace_mode = mo.ui.dropdown(
        options=workspace_mode_options,
        value=workspace_mode_options[0],
        label="Workspace view",
    )
    if eye_angle_run_rows or analysis_dataset_rows or gaze_tracking_rows:
        mode_guidance = mo.md(
            f"Found **{len(analysis_dataset_rows)} analysis-ready track dataset(s)** "
            f"**{len(eye_angle_run_rows)} eye-angle run(s)**, and "
            f"**{len(gaze_tracking_rows)} chaser-gaze component(s)**. Guided mode "
            "presents semantic handles and named traces; Advanced storage exposes "
            "physical groups and arrays."
        )
    else:
        mode_guidance = mo.callout(
            "No supported guided analysis family was discovered. The bounded "
            "physical storage browser is active.",
            kind="info",
        )
    mo.vstack([mo.md("## Choose a view"), workspace_mode, mode_guidance])
    return analysis_dataset_rows, eye_angle_run_rows, gaze_tracking_rows, workspace_mode


@app.cell(hide_code=True)
def _(workspace_mode):
    advanced_storage_mode = workspace_mode.value == "Advanced storage"
    return (advanced_storage_mode,)


@app.cell(hide_code=True)
def _(advanced_storage_mode, gaze_tracking_rows, mo):
    gaze_component_label_to_path = {}
    gaze_component_picker = None
    if not advanced_storage_mode and gaze_tracking_rows:
        gaze_component_label_to_path = {
            (
                f"{row['component_name']} · {row['frame_count']:,} frames"
                + (" · latest" if row["is_latest_complete"] else "")
            ): str(row["component_path"])
            for row in gaze_tracking_rows
        }
        gaze_component_labels = list(gaze_component_label_to_path)
        gaze_component_picker = mo.ui.dropdown(
            options=gaze_component_labels,
            value=gaze_component_labels[0],
            label="Chaser-gaze analysis",
            searchable=True,
        )
        gaze_component_picker_output = mo.vstack(
            [
                mo.md("## Eye–chaser tracking"),
                mo.md(
                    "Select a completed recording-level component. Its small "
                    "epoch summaries and persisted PNG are loaded; framewise "
                    "arrays remain lazy in `exploration`."
                ),
                gaze_component_picker,
            ]
        )
    else:
        gaze_component_picker_output = mo.md("")
    gaze_component_picker_output
    return gaze_component_label_to_path, gaze_component_picker


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    build_chaser_gaze_tracking_output,
    gaze_component_label_to_path,
    gaze_component_picker,
    load_chaser_gaze_tracking_view,
    mo,
    source_path,
):
    gaze_tracking_view = None
    if not advanced_storage_mode and gaze_component_picker is not None:
        try:
            gaze_tracking_view = load_chaser_gaze_tracking_view(
                source_path,
                gaze_component_label_to_path[gaze_component_picker.value],
            )
            gaze_tracking_output = build_chaser_gaze_tracking_output(
                mo,
                loaded=gaze_tracking_view,
            )
        except Exception as exc:
            gaze_tracking_output = mo.callout(
                mo.md(
                    f"Chaser-gaze view failed: `{type(exc).__name__}: {exc}`"
                ),
                kind="danger",
            )
    else:
        gaze_tracking_output = mo.md("")
    gaze_tracking_output
    return (gaze_tracking_view,)


@app.cell(hide_code=True)
def _(mo):
    persisted_png_discover = mo.ui.run_button(
        label="Discover persisted PNGs",
        kind="neutral",
    )
    mo.vstack(
        [
            mo.md("## Persisted visualization gallery"),
            mo.md(
                "Discovery is opt-in because an arbitrary Zarr has no root-level "
                "plot index and may require a metadata walk. No image payload is "
                "read until you select **Load selected PNG**."
            ),
            persisted_png_discover,
        ]
    )
    return (persisted_png_discover,)


@app.cell(hide_code=True)
def _(mo, persisted_png_discover, zarr_workspace):
    persisted_png_rows = []
    persisted_png_picker = None
    if persisted_png_discover.value:
        try:
            persisted_png_rows = zarr_workspace.visualization_artifacts(
                max_artifacts=500
            )
            persisted_png_label_to_path = {
                f"{row['name']} · {row['path']}": str(row["path"])
                for row in persisted_png_rows
            }
            if persisted_png_label_to_path:
                persisted_png_labels = list(persisted_png_label_to_path)
                persisted_png_picker = mo.ui.dropdown(
                    options=persisted_png_labels,
                    value=persisted_png_labels[0],
                    label="Persisted PNG",
                    searchable=True,
                )
                persisted_png_discovery_output = mo.vstack(
                    [
                        mo.md(
                            f"Found **{len(persisted_png_rows):,}** persisted PNG(s)."
                        ),
                        persisted_png_picker,
                        mo.ui.table(
                            persisted_png_rows,
                            selection=None,
                            pagination=True,
                            page_size=15,
                            show_download=False,
                        ),
                    ]
                )
            else:
                persisted_png_discovery_output = mo.callout(
                    "No persisted PNG artifacts were found.", kind="info"
                )
        except Exception as exc:
            persisted_png_label_to_path = {}
            persisted_png_discovery_output = mo.callout(
                mo.md(
                    f"PNG discovery failed: `{type(exc).__name__}: {exc}`"
                ),
                kind="danger",
            )
    else:
        persisted_png_label_to_path = {}
        persisted_png_discovery_output = mo.md("")
    persisted_png_discovery_output
    return persisted_png_label_to_path, persisted_png_picker, persisted_png_rows


@app.cell(hide_code=True)
def _(mo, persisted_png_picker):
    persisted_png_load = mo.ui.run_button(
        label="Load selected PNG",
        kind="success",
        disabled=persisted_png_picker is None,
    )
    persisted_png_load if persisted_png_picker is not None else mo.md("")
    return (persisted_png_load,)


@app.cell(hide_code=True)
def _(
    mo,
    persisted_png_label_to_path,
    persisted_png_load,
    persisted_png_picker,
    png_bytes_to_markdown_image,
    zarr_workspace,
):
    if persisted_png_picker is None or not persisted_png_load.value:
        persisted_png_output = mo.md("")
    else:
        try:
            persisted_png_path, persisted_png_bytes = zarr_workspace.load_png(
                persisted_png_label_to_path[persisted_png_picker.value]
            )
            persisted_png_output = mo.vstack(
                [
                    mo.md(f"### Persisted PNG\n\n`{persisted_png_path}`"),
                    png_bytes_to_markdown_image(
                        mo,
                        persisted_png_bytes,
                        alt_text=persisted_png_path.rsplit("/", 1)[-1],
                    ),
                ]
            )
        except Exception as exc:
            persisted_png_output = mo.callout(
                mo.md(f"PNG load failed: `{type(exc).__name__}: {exc}`"),
                kind="danger",
            )
    persisted_png_output
    return


@app.cell(hide_code=True)
def _(advanced_storage_mode, analysis_dataset_rows, mo):
    analysis_dataset_label_to_id = {}
    analysis_dataset_picker = None
    if not advanced_storage_mode and analysis_dataset_rows:
        analysis_dataset_label_to_id = {
            str(row["label"]): str(row["dataset_id"])
            for row in analysis_dataset_rows
        }
        analysis_dataset_labels = list(analysis_dataset_label_to_id)
        analysis_dataset_picker = mo.ui.dropdown(
            options=analysis_dataset_labels,
            value=analysis_dataset_labels[0],
            label="Analysis-ready dataset",
            searchable=True,
        )
        analysis_dataset_picker_output = mo.vstack(
            [
                mo.md("## Analysis data workspace"),
                mo.md(
                    "Select a scientific dataset rather than a physical Zarr array. "
                    "The resulting `analysis_dataset` handle remains lazy until a "
                    "bounded NumPy or Polars copy is requested."
                ),
                analysis_dataset_picker,
            ]
        )
    else:
        analysis_dataset_picker_output = mo.md("")
    analysis_dataset_picker_output
    return analysis_dataset_label_to_id, analysis_dataset_picker


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    analysis_dataset_label_to_id,
    analysis_dataset_picker,
    analysis_dataset_rows,
    mo,
    zarr_workspace,
):
    analysis_dataset = None
    if not advanced_storage_mode and analysis_dataset_picker is not None:
        analysis_dataset_id = analysis_dataset_label_to_id[
            analysis_dataset_picker.value
        ]
        analysis_dataset_descriptor = next(
            row
            for row in analysis_dataset_rows
            if row["dataset_id"] == analysis_dataset_id
        )
        analysis_dataset = zarr_workspace.dataset(
            analysis_dataset_descriptor
        )
        analysis_dataset_summary = analysis_dataset.summary()
        analysis_dataset_output = mo.vstack(
            [
                mo.tree(
                    {
                        key: value
                        for key, value in analysis_dataset_summary.items()
                        if key
                        not in {
                            "dataset_id",
                            "label",
                            "track_path",
                            "run_path",
                        }
                    },
                    label="Semantic dataset",
                ),
                mo.md(
                    f"Physical value source: `{analysis_dataset.value_path}`. "
                    "Use `analysis_dataset.handles()` for lazy Zarr handles, "
                    "`analysis_dataset.to_numpy(...)` for a writable array copy, or "
                    "`analysis_dataset.to_polars(...)` for analysis-ready columns."
                ),
            ]
        )
    else:
        analysis_dataset_output = mo.md("")
    analysis_dataset_output
    return (analysis_dataset,)


@app.cell(hide_code=True)
def _(advanced_storage_mode, analysis_dataset, mo):
    analysis_copy_scope = None
    analysis_copy_estimated_bytes = 0
    if not advanced_storage_mode and analysis_dataset is not None:
        analysis_copy_estimated_bytes = analysis_dataset.estimated_copy_nbytes()
        analysis_copy_scope = mo.ui.dropdown(
            options=["Complete dataset", "Bounded window"],
            value="Complete dataset",
            label="Working-copy scope",
        )
        analysis_copy_scope_output = mo.vstack(
            [
                analysis_copy_scope,
                mo.md(
                    f"The complete aligned table contains "
                    f"**{analysis_dataset.row_count:,} rows** and is estimated at "
                    f"**{analysis_copy_estimated_bytes / (1024**2):,.1f} MiB** of raw "
                    "column data before small DataFrame/runtime overhead."
                ),
            ]
        )
    else:
        analysis_copy_scope_output = mo.md("")
    analysis_copy_scope_output
    return analysis_copy_estimated_bytes, analysis_copy_scope


@app.cell(hide_code=True)
def _(advanced_storage_mode, analysis_copy_scope, analysis_dataset, mo):
    analysis_copy_start = None
    analysis_copy_stop = None
    analysis_copy_stride = None
    analysis_copy_run = None
    if (
        not advanced_storage_mode
        and analysis_dataset is not None
        and analysis_copy_scope is not None
    ):
        analysis_copy_is_complete = (
            analysis_copy_scope.value == "Complete dataset"
        )
        analysis_copy_start = mo.ui.number(
            start=0,
            stop=max(0, analysis_dataset.row_count - 1),
            step=1,
            value=0,
            label="Start row",
            disabled=analysis_copy_is_complete,
        )
        analysis_copy_stop = mo.ui.number(
            start=1,
            stop=analysis_dataset.row_count,
            step=1,
            value=min(analysis_dataset.row_count, 1_800),
            label="Stop row (exclusive)",
            disabled=analysis_copy_is_complete,
        )
        analysis_copy_stride = mo.ui.number(
            start=1,
            stop=1_000,
            step=1,
            value=1,
            label="Row stride",
            disabled=analysis_copy_is_complete,
        )
        analysis_copy_run = mo.ui.run_button(
            label=(
                "Load complete dataset"
                if analysis_copy_is_complete
                else "Load bounded working copy"
            ),
            kind="success",
        )
        analysis_copy_controls_output = mo.vstack(
            [
                mo.hstack(
                    [
                        analysis_copy_start,
                        analysis_copy_stop,
                        analysis_copy_stride,
                        analysis_copy_run,
                    ],
                    justify="start",
                    gap=1,
                    wrap=True,
                ),
                mo.md(
                    (
                        "The complete copy has a 1 GB raw-column memory guard. "
                        if analysis_copy_is_complete
                        else "A bounded copy is limited to 100,000 source rows. "
                    )
                    + "It is detached from the read-only Zarr and becomes available "
                    "below as the Polars DataFrame `analysis_data`."
                ),
            ]
        )
    else:
        analysis_copy_controls_output = mo.md("")
    analysis_copy_controls_output
    return (
        analysis_copy_run,
        analysis_copy_start,
        analysis_copy_stop,
        analysis_copy_stride,
    )


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    analysis_copy_estimated_bytes,
    analysis_copy_run,
    analysis_copy_scope,
    analysis_copy_start,
    analysis_copy_stop,
    analysis_copy_stride,
    analysis_dataset,
    mo,
):
    analysis_data = None
    if advanced_storage_mode or analysis_dataset is None:
        analysis_copy_output = mo.md("")
    elif analysis_copy_run is None or not analysis_copy_run.value:
        analysis_copy_output = mo.md(
            "Select the load button when you want a Polars object for custom cells. "
            "Metadata and lazy handles above require no value read."
        )
    else:
        try:
            if analysis_copy_scope.value == "Complete dataset":
                analysis_data = analysis_dataset.to_polars_full(
                    max_copy_bytes=1_000_000_000,
                )
            else:
                analysis_data = analysis_dataset.to_polars(
                    start=int(analysis_copy_start.value or 0),
                    stop=int(analysis_copy_stop.value or 0),
                    stride=int(analysis_copy_stride.value or 1),
                    max_source_rows=100_000,
                )
            analysis_copy_output = mo.vstack(
                [
                    mo.callout(
                        f"Copied {analysis_data.height:,} row(s) and "
                        f"{analysis_data.width:,} column(s) into memory. The source "
                        f"Zarr was not modified. Estimated raw columns: "
                        f"{analysis_copy_estimated_bytes / (1024**2):,.1f} MiB.",
                        kind="success",
                    ),
                    mo.ui.table(
                        analysis_data.head(25),
                        selection=None,
                        pagination=True,
                        page_size=25,
                        show_download=False,
                        label="First 25 copied rows",
                    ),
                ]
            )
        except Exception as exc:
            analysis_copy_output = mo.callout(
                mo.md(
                    f"Working copy refused: `{type(exc).__name__}: {exc}`"
                ),
                kind="danger",
            )
    analysis_copy_output
    return (analysis_data,)


@app.cell(hide_code=True)
def _(advanced_storage_mode, eye_angle_run_rows, mo):
    guided_run_label_to_path = {}
    guided_run_picker = None
    if not advanced_storage_mode and eye_angle_run_rows:
        guided_run_label_to_path = {
            (
                f"{row['run_name']} · {row['status'] or 'status unknown'} · "
                f"{row['frame_count']:,} frames"
            ): str(row["run_path"])
            for row in eye_angle_run_rows
            if row.get("frame_angles_path")
        }
        guided_run_labels = list(guided_run_label_to_path)
        complete_labels = [
            label
            for label in guided_run_labels
            if next(
                row
                for row in eye_angle_run_rows
                if row["run_path"] == guided_run_label_to_path[label]
            )["status"].casefold()
            == "complete"
        ]
        guided_default_run_label = (
            complete_labels[-1] if complete_labels else guided_run_labels[-1]
        )
        guided_run_picker = mo.ui.dropdown(
            options=guided_run_labels,
            value=guided_default_run_label,
            label="Eye-angle analysis run",
            searchable=True,
        )
        guided_run_picker_output = mo.vstack(
            [mo.md("## Eye angles and convergence"), guided_run_picker]
        )
    else:
        guided_run_picker_output = mo.md("")
    guided_run_picker_output
    return guided_run_label_to_path, guided_run_picker


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    eye_angle_run_rows,
    guided_run_label_to_path,
    guided_run_picker,
    mo,
    zarr_workspace,
):
    guided_array_path = ""
    guided_channel_rows = []
    guided_coordinate_summary = None
    guided_representation_label_to_value = {}
    guided_representation_picker = None
    if not advanced_storage_mode and guided_run_picker is not None:
        guided_run_path = guided_run_label_to_path[guided_run_picker.value]
        guided_run_row = next(
            row for row in eye_angle_run_rows if row["run_path"] == guided_run_path
        )
        guided_array_path = str(guided_run_row["frame_angles_path"])
        guided_channel_rows = [
            row
            for row in zarr_workspace.channel_index(guided_array_path)
            if not row.get("compatibility_alias_of")
        ]
        guided_coordinate_summary = zarr_workspace.coordinate_summary(
            guided_array_path
        )
        guided_representation_order = [
            "eye_frame",
            "gaze",
            "nasal_gaze",
            "major",
            "centroid",
            "legacy",
        ]
        guided_available_representations = {
            str(row.get("representation") or "all")
            for row in guided_channel_rows
        }
        guided_representation_values = [
            value
            for value in guided_representation_order
            if value in guided_available_representations
        ]
        guided_representation_values.extend(
            sorted(
                guided_available_representations
                - set(guided_representation_values)
            )
        )
        guided_representation_labels = {
            "eye_frame": "Per-eye angles and convergence",
            "gaze": "Gaze direction",
            "nasal_gaze": "Nasal-gaze convergence",
            "major": "Major-axis orientation",
            "centroid": "Centroid diagnostics",
            "legacy": "Legacy compatibility",
            "legacy_minor": "Legacy minor-axis compatibility",
            "all": "Available channels",
        }
        guided_representation_label_to_value = {
            guided_representation_labels.get(value, value.replace("_", " ").title()): value
            for value in guided_representation_values
        }
        guided_run_summary = {
            key: value
            for key, value in guided_run_row.items()
            if key not in {"run_path", "frame_angles_path"}
        }
        guided_representation_options = list(guided_representation_label_to_value)
        if guided_representation_options:
            guided_representation_picker = mo.ui.dropdown(
                options=guided_representation_options,
                value=guided_representation_options[0],
                label="Scientific representation",
            )
            guided_representation_output = mo.vstack(
                [
                    mo.hstack(
                        [
                            guided_representation_picker,
                            mo.tree(guided_run_summary, label="Run summary"),
                        ],
                        justify="start",
                        gap=1,
                        widths=[1, 2],
                    ),
                    mo.md(
                        "Representations are resolved from the persisted channel "
                        "index; support arrays, indexes, and compatibility aliases "
                        "are not shown as separate datasets here."
                    ),
                ]
            )
        else:
            guided_representation_output = mo.callout(
                "This run has a dense frame-angle array but no readable semantic "
                "channel index. Use Advanced storage to inspect its contract.",
                kind="warn",
            )
    else:
        guided_representation_output = mo.md("")
    guided_representation_output
    return (
        guided_array_path,
        guided_channel_rows,
        guided_coordinate_summary,
        guided_representation_label_to_value,
        guided_representation_picker,
    )


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    guided_channel_rows,
    guided_representation_label_to_value,
    guided_representation_picker,
    mo,
):
    guided_channel_label_to_index = {}
    guided_channel_picker = None
    guided_selected_representation = ""
    if not advanced_storage_mode and guided_representation_picker is not None:
        guided_selected_representation = guided_representation_label_to_value[
            guided_representation_picker.value
        ]
        guided_representation_channels = [
            row
            for row in guided_channel_rows
            if str(row.get("representation") or "all")
            == guided_selected_representation
        ]
        guided_channel_labels = [
            (
                f"{str(row.get('eye') or 'derived').title()} · {row['name']}"
                + (f" [{row['units']}]" if row.get("units") else "")
            )
            for row in guided_representation_channels
        ]
        guided_channel_label_to_index = {
            label: int(row["index"])
            for label, row in zip(
                guided_channel_labels,
                guided_representation_channels,
                strict=True,
            )
        }
        guided_preferred_names = {
            "eye_frame": (
                "left_eye_angle_deg_smoothed",
                "right_eye_angle_deg_smoothed",
                "vergence_eye_angle_deg_smoothed",
            ),
            "gaze": (
                "left_gaze_signed_deg_smoothed",
                "right_gaze_signed_deg_smoothed",
                "vergence_gaze_signed_deg_smoothed",
            ),
            "nasal_gaze": (
                "left_nasal_gaze_deg_smoothed",
                "right_nasal_gaze_deg_smoothed",
                "mean_eye_vergence_gaze_deg_smoothed",
            ),
            "major": (
                "left_major_signed_deg_smoothed",
                "right_major_signed_deg_smoothed",
                "vergence_major_signed_deg_smoothed",
            ),
            "centroid": (
                "left_centroid_deg_smoothed",
                "right_centroid_deg_smoothed",
                "vergence_centroid_deg_smoothed",
            ),
            "legacy": (
                "left_minor_signed_deg_smoothed",
                "right_minor_signed_deg_smoothed",
                "vergence_minor_signed_deg_smoothed",
            ),
        }
        preferred_for_representation = guided_preferred_names.get(
            guided_selected_representation, ()
        )
        guided_default_channel_labels = [
            label
            for label, row in zip(
                guided_channel_labels,
                guided_representation_channels,
                strict=True,
            )
            if row["name"] in preferred_for_representation
        ][:3]
        if not guided_default_channel_labels:
            guided_default_channel_labels = [
                label
                for label, row in zip(
                    guided_channel_labels,
                    guided_representation_channels,
                    strict=True,
                )
                if "smoothed" in str(row["name"])
            ][:3]
        if not guided_default_channel_labels:
            guided_default_channel_labels = guided_channel_labels[:3]
        guided_channel_picker = mo.ui.multiselect(
            options=guided_channel_labels,
            value=guided_default_channel_labels,
            label="Traces",
            max_selections=6,
        )
        guided_channel_table_rows = [
            {
                "name": row["name"],
                "eye": row.get("eye", ""),
                "measurement": row.get("value_kind", ""),
                "units": row.get("units", ""),
                "formula": row.get("formula", ""),
            }
            for row in guided_representation_channels
        ]
        guided_channel_output = mo.vstack(
            [
                guided_channel_picker,
                mo.accordion(
                    {
                        "Available named channels": mo.ui.table(
                            guided_channel_table_rows,
                            selection=None,
                            pagination=True,
                            page_size=12,
                            show_download=False,
                        )
                    }
                ),
            ]
        )
    else:
        guided_channel_output = mo.md("")
    guided_channel_output
    return (
        guided_channel_label_to_index,
        guided_channel_picker,
        guided_selected_representation,
    )


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    analysis_dataset,
    guided_array_path,
    guided_coordinate_summary,
    mo,
    zarr_workspace,
):
    guided_window_start = None
    guided_window_stop = None
    guided_max_points = None
    guided_plot_run = None
    guided_window_is_seconds = False
    if not advanced_storage_mode and guided_array_path:
        guided_array_info = zarr_workspace.info(guided_array_path)
        guided_frame_count = int(guided_array_info["shape"][0])
        if (
            guided_coordinate_summary is not None
            and guided_coordinate_summary.get("sample_interval_seconds")
        ):
            guided_window_is_seconds = True
            guided_window_min = float(
                guided_coordinate_summary["start_seconds"]
            )
            guided_window_max = float(guided_coordinate_summary["stop_seconds"])
            guided_window_step = max(
                0.001,
                float(guided_coordinate_summary["sample_interval_seconds"]),
            )
            guided_window_max += guided_window_step
            guided_window_default_stop = min(
                guided_window_max, guided_window_min + 60.0
            )
            guided_window_start_label = "Start time (s)"
            guided_window_stop_label = "Stop time (s)"
        else:
            guided_window_min = 0
            guided_window_max = guided_frame_count
            guided_window_step = 1
            guided_window_default_stop = min(guided_frame_count, 1_800)
            guided_window_start_label = "Start row"
            guided_window_stop_label = "Stop row (exclusive)"
        guided_window_start = mo.ui.number(
            start=guided_window_min,
            stop=guided_window_max,
            step=guided_window_step,
            value=guided_window_min,
            label=guided_window_start_label,
        )
        guided_window_stop = mo.ui.number(
            start=guided_window_min,
            stop=guided_window_max,
            step=guided_window_step,
            value=guided_window_default_stop,
            label=guided_window_stop_label,
        )
        guided_max_points = mo.ui.number(
            start=100,
            stop=20_000,
            step=100,
            value=5_000,
            label="Maximum plotted points",
        )
        guided_plot_run = mo.ui.run_button(
            label="Plot selected eye traces", kind="success"
        )
        guided_window_output = mo.vstack(
            [
                mo.hstack(
                    [
                        guided_window_start,
                        guided_window_stop,
                        guided_max_points,
                        guided_plot_run,
                    ],
                    justify="start",
                    gap=1,
                    wrap=True,
                ),
                mo.md(
                    "The default is the first 60 seconds. Each interaction is limited "
                    "to 100,000 source frames, and the rendered trace is decimated to "
                    "the selected point limit."
                ),
            ]
        )
    else:
        guided_window_output = mo.md("")
    guided_window_output
    return (
        guided_max_points,
        guided_plot_run,
        guided_window_is_seconds,
        guided_window_start,
        guided_window_stop,
    )


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    go,
    guided_array_path,
    guided_channel_label_to_index,
    guided_channel_picker,
    guided_coordinate_summary,
    guided_max_points,
    guided_plot_run,
    guided_selected_representation,
    guided_window_is_seconds,
    guided_window_start,
    guided_window_stop,
    mo,
    np,
    zarr_workspace,
):
    if advanced_storage_mode or not guided_array_path:
        guided_trace_output = mo.md("")
    elif guided_plot_run is None or not guided_plot_run.value:
        guided_trace_output = mo.md(
            "Select **Plot selected eye traces** to read the bounded time window."
        )
    elif guided_channel_picker is None or not guided_channel_picker.value:
        guided_trace_output = mo.callout(
            "Select at least one named trace.", kind="warn"
        )
    else:
        try:
            if guided_window_is_seconds:
                coordinate_start = float(
                    guided_coordinate_summary["start_seconds"]
                )
                coordinate_interval = float(
                    guided_coordinate_summary["sample_interval_seconds"]
                )
                guided_start_row = max(
                    0,
                    int(
                        np.floor(
                            (float(guided_window_start.value) - coordinate_start)
                            / coordinate_interval
                        )
                    ),
                )
                guided_stop_row = min(
                    int(guided_coordinate_summary["row_count"]),
                    int(
                        np.ceil(
                            (float(guided_window_stop.value) - coordinate_start)
                            / coordinate_interval
                        )
                    ),
                )
            else:
                guided_start_row = int(guided_window_start.value or 0)
                guided_stop_row = int(guided_window_stop.value or 0)
            if guided_stop_row <= guided_start_row:
                raise ValueError("Stop must be greater than start.")

            guided_trace_figure = go.Figure()
            for guided_trace_label in guided_channel_picker.value:
                guided_trace_frame = zarr_workspace.trace_frame(
                    guided_array_path,
                    column=guided_channel_label_to_index[guided_trace_label],
                    start=guided_start_row,
                    stop=guided_stop_row,
                    max_points=int(guided_max_points.value or 5_000),
                    max_source_rows=100_000,
                    coordinate_path=(
                        str(guided_coordinate_summary["path"])
                        if guided_coordinate_summary is not None
                        else None
                    ),
                )
                guided_x_column = (
                    "time_seconds"
                    if "time_seconds" in guided_trace_frame.columns
                    else "row_index"
                )
                guided_trace_figure.add_trace(
                    go.Scattergl(
                        x=guided_trace_frame[guided_x_column].to_numpy(),
                        y=guided_trace_frame["value"].to_numpy(),
                        mode="lines",
                        name=str(guided_trace_label),
                        hovertemplate=(
                            "%{x:.3f} s<br>%{y:.3f}<extra>%{fullData.name}</extra>"
                            if guided_x_column == "time_seconds"
                            else "row %{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
            guided_trace_figure.update_layout(
                title=(
                    "Eye-angle traces · "
                    f"{guided_selected_representation.replace('_', ' ')}"
                ),
                xaxis_title=(
                    "Time (s)"
                    if guided_coordinate_summary is not None
                    else "Source row index"
                ),
                yaxis_title="Angle (deg)",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Named trace",
                height=520,
            )
            guided_trace_output = mo.ui.plotly(
                guided_trace_figure,
                config={"displaylogo": False, "scrollZoom": True},
            )
        except Exception as exc:
            guided_trace_output = mo.callout(
                mo.md(
                    f"Eye-angle plot refused: `{type(exc).__name__}: {exc}`"
                ),
                kind="danger",
            )
    guided_trace_output
    return


@app.cell(hide_code=True)
def _(advanced_storage_mode, mo):
    inventory_path = mo.ui.text(
        value="", label="Group path", placeholder="root (leave blank)"
    )
    inventory_depth = mo.ui.number(
        start=0, stop=8, step=1, value=2, label="Traversal depth"
    )
    inventory_limit = mo.ui.number(
        start=1, stop=2_000, step=25, value=250, label="Maximum nodes"
    )
    if advanced_storage_mode:
        inventory_controls_output = mo.vstack(
            [
                mo.md("## Advanced physical storage"),
                mo.callout(
                    "This view exposes implementation arrays, indexes, support "
                    "coordinates, and compatibility surfaces. Use it when the guided "
                    "scientific adapter does not cover the question.",
                    kind="info",
                ),
                mo.hstack(
                    [inventory_path, inventory_depth, inventory_limit],
                    justify="start",
                    gap=1,
                    wrap=True,
                ),
            ]
        )
    else:
        inventory_controls_output = mo.md("")
    inventory_controls_output
    return inventory_depth, inventory_limit, inventory_path


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    inventory_depth,
    inventory_limit,
    inventory_path,
    mo,
    zarr_workspace,
):
    if not advanced_storage_mode:
        inventory_rows = []
        inventory_table = None
        inventory_output = mo.md("")
    else:
        try:
            inventory_rows = zarr_workspace.walk(
                inventory_path.value,
                max_depth=int(inventory_depth.value or 0),
                max_items=int(inventory_limit.value or 250),
            )
            inventory_display_rows = [
                {
                    **row,
                    "shape": str(row.get("shape", "")),
                    "chunks": str(row.get("chunks", "")),
                }
                for row in inventory_rows
            ]
            inventory_table = mo.ui.table(
                inventory_display_rows,
                selection="single",
                pagination=True,
                page_size=25,
                show_download=False,
                max_height=600,
                label="Select one group or array",
            )
            inventory_output = mo.vstack(
                [
                    mo.md(f"## Metadata inventory · {len(inventory_rows):,} node(s)"),
                    mo.md(
                        "Select a row to reveal its metadata and make its relative "
                        "path available to the preview and exploration cells below."
                    ),
                    inventory_table,
                ]
            )
        except Exception as exc:
            inventory_rows = []
            inventory_table = None
            inventory_output = mo.callout(
                mo.md(f"Could not inspect that group: `{type(exc).__name__}: {exc}`"),
                kind="danger",
            )
    inventory_output
    return inventory_rows, inventory_table


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    inventory_depth,
    inventory_limit,
    inventory_table,
    mo,
    zarr_workspace,
):
    inventory_selected_rows = (
        inventory_table.value if inventory_table is not None else []
    )
    inventory_selected_path = (
        str(inventory_selected_rows[0]["path"]) if inventory_selected_rows else ""
    )
    inventory_selected_kind = ""
    group_contents_table = None
    if inventory_selected_path:
        try:
            inventory_selected_info = zarr_workspace.info(inventory_selected_path)
            inventory_selected_kind = str(
                inventory_selected_info.get("kind", "")
            )
            inventory_selected_attrs = zarr_workspace.attrs(inventory_selected_path)
            selection_parts = inventory_selected_path.split("/")
            selection_breadcrumb = " › ".join(["/"] + selection_parts)
            selection_panels = [
                mo.md(
                    f"### Inventory selection: `{inventory_selected_path}`\n\n"
                    f"**Path:** {selection_breadcrumb}"
                ),
                mo.hstack(
                    [
                        mo.tree(inventory_selected_info, label="Metadata"),
                        mo.tree(inventory_selected_attrs, label="Attributes"),
                    ],
                    justify="start",
                    gap=1,
                    widths="equal",
                ),
            ]
            if inventory_selected_kind == "group":
                group_contents_rows = zarr_workspace.walk(
                    inventory_selected_path,
                    max_depth=max(1, int(inventory_depth.value or 2)),
                    max_items=int(inventory_limit.value or 250),
                )
                group_prefix = f"{inventory_selected_path}/"
                group_contents_display_rows = [
                    {
                        **row,
                        "name": str(row["path"]).rsplit("/", 1)[-1],
                        "relative_path": str(row["path"])[len(group_prefix) :],
                        "shape": str(row.get("shape", "")),
                        "chunks": str(row.get("chunks", "")),
                    }
                    for row in group_contents_rows
                ]
                if group_contents_display_rows:
                    group_contents_table = mo.ui.table(
                        group_contents_display_rows,
                        selection="single",
                        pagination=True,
                        page_size=25,
                        show_download=False,
                        max_height=500,
                        label="Select a descendant group or array",
                    )
                    selection_panels.extend(
                        [
                            mo.md(
                                f"#### Contents of `{inventory_selected_path}` · "
                                f"{len(group_contents_rows):,} bounded descendant(s)"
                            ),
                            mo.md(
                                "This second table reveals the selected group's "
                                "contents. Selecting one of its rows updates the "
                                "active dataset used by preview and `selected_path`."
                            ),
                            group_contents_table,
                        ]
                    )
                else:
                    selection_panels.append(
                        mo.callout("This group has no discoverable child nodes.", kind="info")
                    )
            inventory_selection_output = mo.vstack(selection_panels)
        except Exception as exc:
            inventory_selection_output = mo.callout(
                mo.md(f"Could not reveal the selection: `{type(exc).__name__}: {exc}`"),
                kind="danger",
            )
    elif advanced_storage_mode:
        inventory_selection_output = mo.md(
            "Select a dataset row above to reveal its metadata and contents."
        )
    else:
        inventory_selection_output = mo.md("")
    inventory_selection_output
    return group_contents_table, inventory_selected_kind, inventory_selected_path


@app.cell(hide_code=True)
def _(
    group_contents_table,
    inventory_selected_kind,
    inventory_selected_path,
    mo,
    zarr_workspace,
):
    contents_selected_rows = (
        group_contents_table.value if group_contents_table is not None else []
    )
    physical_selected_path = (
        str(contents_selected_rows[0]["path"])
        if contents_selected_rows
        else inventory_selected_path
    )
    physical_selected_kind = inventory_selected_kind
    if physical_selected_path:
        try:
            physical_selected_info = zarr_workspace.info(physical_selected_path)
            physical_selected_kind = str(physical_selected_info.get("kind", ""))
            physical_selected_attrs = zarr_workspace.attrs(physical_selected_path)
            active_source = (
                "group contents table"
                if contents_selected_rows
                else "main inventory table"
            )
            if physical_selected_kind == "group":
                active_guidance = mo.callout(
                    "The active node is a group. Select one of its descendant "
                    "array rows in the contents table to preview values.",
                    kind="info",
                )
            else:
                active_guidance = mo.md("")
            active_selection_output = mo.vstack(
                [
                    mo.md(
                        f"### Active dataset: `{physical_selected_path}`\n\n"
                        f"Selected from the {active_source}."
                    ),
                    active_guidance,
                    mo.hstack(
                        [
                            mo.tree(physical_selected_info, label="Active metadata"),
                            mo.tree(physical_selected_attrs, label="Active attributes"),
                        ],
                        justify="start",
                        gap=1,
                        widths="equal",
                    ),
                ]
            )
        except Exception as exc:
            physical_selected_kind = ""
            active_selection_output = mo.callout(
                mo.md(f"Could not activate the selection: `{type(exc).__name__}: {exc}`"),
                kind="danger",
            )
    else:
        active_selection_output = mo.md("")
    active_selection_output
    return physical_selected_kind, physical_selected_path


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    guided_array_path,
    physical_selected_kind,
    physical_selected_path,
):
    if advanced_storage_mode:
        selected_path = physical_selected_path
        selected_kind = physical_selected_kind
    else:
        selected_path = (
            analysis_dataset.value_path
            if analysis_dataset is not None
            else guided_array_path
        )
        selected_kind = "array" if selected_path else ""
    return selected_kind, selected_path


@app.cell(hide_code=True)
def _(advanced_storage_mode, mo, selected_kind, selected_path):
    selected_is_array = selected_kind == "array"
    preview_rows = mo.ui.number(
        start=0,
        stop=10_000,
        step=10,
        value=100,
        label="Leading rows",
        disabled=not selected_is_array,
    )
    preview_run = mo.ui.run_button(
        label="Load selected array preview",
        kind="success",
        disabled=not selected_is_array,
    )
    if selected_kind == "group":
        preview_instruction = (
            "The selected path is a group. Select an `array` row to enable preview."
        )
    elif not selected_path:
        preview_instruction = "Select an `array` row above to enable preview."
    else:
        preview_instruction = (
            "Request a small leading-axis preview. Large multidimensional rows may "
            "still exceed the 100,000-element guard; use an explicit tuple of slices "
            "in the exploration cell below."
        )
    if advanced_storage_mode:
        preview_controls_output = mo.vstack(
            [
                mo.md("## Array preview"),
                mo.md(
                    f"Selected path: `{selected_path or 'none'}`. "
                    f"{preview_instruction}"
                ),
                mo.hstack(
                    [preview_rows, preview_run],
                    justify="start",
                    gap=1,
                    wrap=True,
                ),
            ]
        )
    else:
        preview_controls_output = mo.md("")
    preview_controls_output
    return preview_rows, preview_run


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    mo,
    preview_rows,
    preview_run,
    selected_kind,
    selected_path,
    zarr_workspace,
):
    if not advanced_storage_mode:
        preview_output = mo.md("")
    elif not selected_path:
        preview_output = mo.callout("Select a dataset row first.", kind="warn")
    elif selected_kind != "array":
        preview_output = mo.callout(
            "The selected node is a group. Select an array row to preview values.",
            kind="info",
        )
    elif not preview_run.value:
        preview_output = mo.md("Select **Load selected array preview** to read values.")
    else:
        try:
            preview_metadata = zarr_workspace.info(selected_path)
            preview_values = zarr_workspace.head(
                selected_path,
                rows=int(preview_rows.value or 0),
            )
            preview_output = mo.vstack(
                [
                    mo.tree(preview_metadata, label="Array metadata"),
                    mo.plain(preview_values),
                ]
            )
        except Exception as exc:
            preview_output = mo.callout(
                mo.md(f"Preview refused: `{type(exc).__name__}: {exc}`"),
                kind="danger",
            )
    preview_output
    return


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    mo,
    np,
    selected_kind,
    selected_path,
    zarr_workspace,
):
    trace_supported = False
    trace_channel_lookup = {}
    trace_channel_picker = None
    trace_start = None
    trace_stop = None
    trace_max_points = None
    trace_run = None
    trace_coordinate_path = None
    if advanced_storage_mode and selected_kind == "array":
        try:
            trace_info = zarr_workspace.info(selected_path)
            trace_shape = tuple(trace_info.get("shape", ()))
            trace_dtype = np.dtype(str(trace_info.get("dtype", "")))
            trace_supported = (
                len(trace_shape) in {1, 2}
                and trace_shape[0] > 0
                and np.issubdtype(trace_dtype, np.number)
            )
        except (TypeError, ValueError):
            trace_shape = ()
            trace_supported = False
    else:
        trace_shape = ()

    if trace_supported:
        trace_channels = zarr_workspace.channel_index(selected_path)
        if len(trace_shape) == 1:
            trace_channel_labels = ["value"]
            trace_channel_lookup = {"value": None}
            trace_default_labels = ["value"]
        elif trace_channels:
            trace_channel_labels = [
                (
                    f"{row['index']}: {row['name']}"
                    + (f" [{row['units']}]" if row.get("units") else "")
                )
                for row in trace_channels
            ]
            trace_channel_lookup = {
                label: int(row["index"])
                for label, row in zip(
                    trace_channel_labels, trace_channels, strict=True
                )
            }
            preferred_names = (
                "left_eye_angle_deg_smoothed",
                "right_eye_angle_deg_smoothed",
                "vergence_eye_angle_deg_smoothed",
            )
            trace_default_labels = [
                label
                for label, row in zip(
                    trace_channel_labels, trace_channels, strict=True
                )
                if row["name"] in preferred_names
            ][:3]
            if not trace_default_labels:
                trace_default_labels = trace_channel_labels[:1]
        else:
            trace_channel_labels = [
                f"column {index}" for index in range(trace_shape[1])
            ]
            trace_channel_lookup = {
                label: index for index, label in enumerate(trace_channel_labels)
            }
            trace_default_labels = trace_channel_labels[:1]

        trace_channel_picker = mo.ui.multiselect(
            options=trace_channel_labels,
            value=trace_default_labels,
            label="Trace channels",
            max_selections=6,
        )
        trace_start = mo.ui.number(
            start=0,
            stop=max(0, trace_shape[0] - 1),
            step=1,
            value=0,
            label="Start row",
        )
        trace_stop = mo.ui.number(
            start=1,
            stop=trace_shape[0],
            step=1,
            value=min(trace_shape[0], 1_800),
            label="Stop row (exclusive)",
        )
        trace_max_points = mo.ui.number(
            start=100,
            stop=20_000,
            step=100,
            value=5_000,
            label="Maximum plotted points",
        )
        trace_run = mo.ui.run_button(label="Plot selected trace", kind="success")
        trace_coordinate_path = zarr_workspace.suggested_coordinate_path(selected_path)
        trace_controls_output = mo.vstack(
            [
                mo.md("## Numeric trace plot"),
                mo.md(
                    "Plot one or more named channels over a bounded row window. "
                    "For compact eye-angle arrays, channel names come from the "
                    "persisted channel index and frame time is selected automatically."
                ),
                mo.hstack(
                    [
                        trace_channel_picker,
                        trace_start,
                        trace_stop,
                        trace_max_points,
                        trace_run,
                    ],
                    justify="start",
                    gap=1,
                    wrap=True,
                ),
                mo.md(
                    f"Coordinate: `{trace_coordinate_path or 'row_index'}`. The "
                    "interactive source-window limit is 100,000 rows; plotted-point "
                    "decimation does not make a wider source scan inexpensive."
                ),
            ]
        )
    else:
        trace_controls_output = mo.md("")
    trace_controls_output
    return (
        trace_channel_lookup,
        trace_channel_picker,
        trace_coordinate_path,
        trace_max_points,
        trace_run,
        trace_start,
        trace_stop,
        trace_supported,
    )


@app.cell(hide_code=True)
def _(
    advanced_storage_mode,
    go,
    mo,
    selected_path,
    trace_channel_lookup,
    trace_channel_picker,
    trace_coordinate_path,
    trace_max_points,
    trace_run,
    trace_start,
    trace_stop,
    trace_supported,
    zarr_workspace,
):
    if advanced_storage_mode is False:
        trace_plot_output = mo.md("")
    elif not trace_supported or trace_run is None:
        trace_plot_output = mo.md("")
    elif not trace_run.value:
        trace_plot_output = mo.md("Select **Plot selected trace** to read the window.")
    elif trace_channel_picker is None or not trace_channel_picker.value:
        trace_plot_output = mo.callout(
            "Select at least one trace channel.", kind="warn"
        )
    else:
        try:
            trace_figure = go.Figure()
            trace_start_row = int(trace_start.value or 0)
            trace_stop_row = int(trace_stop.value or 0)
            trace_point_limit = int(trace_max_points.value or 5_000)
            for trace_label in trace_channel_picker.value:
                trace_frame = zarr_workspace.trace_frame(
                    selected_path,
                    column=trace_channel_lookup[trace_label],
                    start=trace_start_row,
                    stop=trace_stop_row,
                    max_points=trace_point_limit,
                    coordinate_path=trace_coordinate_path,
                )
                trace_x_column = (
                    "time_seconds"
                    if "time_seconds" in trace_frame.columns
                    else "row_index"
                )
                trace_figure.add_trace(
                    go.Scattergl(
                        x=trace_frame[trace_x_column].to_numpy(),
                        y=trace_frame["value"].to_numpy(),
                        mode="lines",
                        name=str(trace_label),
                        hovertemplate=(
                            "%{x:.3f} s<br>%{y:.3f}<extra>%{fullData.name}</extra>"
                            if trace_x_column == "time_seconds"
                            else "row %{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
            trace_figure.update_layout(
                title=f"{selected_path} · rows {trace_start_row:,}–{trace_stop_row:,}",
                xaxis_title=(
                    "Time (s)" if trace_coordinate_path else "Source row index"
                ),
                yaxis_title="Value",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Channel",
                height=520,
            )
            trace_plot_output = mo.ui.plotly(
                trace_figure,
                config={"displaylogo": False, "scrollZoom": True},
            )
        except Exception as exc:
            trace_plot_output = mo.callout(
                mo.md(f"Trace plot refused: `{type(exc).__name__}: {exc}`"),
                kind="danger",
            )
    trace_plot_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("---\n\n## Exploration API for people and Pair agents"),
            mo.callout(
                mo.md(
                    "Start from the visible `exploration` object below. Prefer metadata "
                    "calls before reads, keep selections explicit, and save any derived "
                    "outputs only under `/workspace`. Do not reopen the source in a "
                    "writable mode."
                ),
                kind="info",
            ),
            mo.md(
                """
```python
exploration.summary()                         # small root summary
selected_path                                 # row selected in the inventory UI
exploration.ls("", max_items=100)            # direct children, metadata only
exploration.walk("tracks", max_depth=2)      # bounded recursive inventory
exploration.find("speed")                     # bounded path search
exploration.info("tracks/speed")              # shape, dtype, chunks, size
exploration.attrs("tracks")                   # bounded attributes
exploration.analysis_datasets()                # semantic track-data catalog
speed = exploration.select_dataset(
    "speed", variant="smoothed", units="mm/s", track_id=0
)
exploration.eye_angle_runs()                   # guided run discovery, metadata only
exploration.channel_index(selected_path)       # named compact-dense columns
exploration.coordinate_summary(selected_path)  # bounded matching-time summary
exploration.visualization_artifacts()           # opt-in persisted-PNG metadata walk
resolved_path, png = exploration.load_png(
    "analysis/.../visualizations/example_summary_png"
)

# When present, the guided chaser-gaze selector exposes small Polars summaries
# and the already-loaded persisted PNG without reading framewise gaze arrays.
gaze_tracking_view.recording_summary_df
gaze_tracking_view.object_vs_virtual_df
gaze_tracking_view.summary_png_bytes

# The guided selector exposes one semantic read-only handle. Array handles do
# not read values; NumPy and Polars methods return detached working copies.
analysis_dataset.summary()
arrays = analysis_dataset.handles()
writable_values = analysis_dataset.to_numpy(start=0, stop=1_800)
working_frame = analysis_dataset.to_polars(start=0, stop=1_800)
complete_frame = analysis_dataset.to_polars_full()  # guarded at 1 GB raw bytes

# Example: contiguous finite speed <= 1 mm/s for at least 30 seconds.
stationary_periods = (
    complete_frame.sort("time_s")
    .with_columns(
        (
            pl.col("speed_mm_s").is_finite()
            & (pl.col("speed_mm_s") <= 1.0)
        ).alias("stationary")
    )
    .with_columns(pl.col("stationary").rle_id().alias("segment_id"))
    .filter(pl.col("stationary"))
    .group_by("segment_id", maintain_order=True)
    .agg(
        pl.col("time_s").first().alias("start_s"),
        pl.col("time_s").last().alias("stop_s"),
    )
    .with_columns((pl.col("stop_s") - pl.col("start_s")).alias("duration_s"))
    .filter(pl.col("duration_s") >= 30.0)
)

# Process a complete long recording without one giant browser allocation.
for batch in analysis_dataset.iter_polars(batch_rows=100_000):
    pass  # aggregate or write derived results beneath /workspace

# `analysis_data` is the optional Polars copy loaded by the controls above.
# Saving it creates a new derived file; it never writes back into the source.
if analysis_data is not None:
    analysis_data.write_parquet("/workspace/my_analysis_copy.parquet")

# Explicit bounded NumPy reads (integers, slices, and ellipsis only):
speed = exploration.read("tracks/speed", slice(0, 1_000))
crop = exploration.read(
    "images/frames", (0, slice(100, 200), slice(100, 200))
)

# Load sibling 1D arrays directly into Polars without Pandas/PyArrow bridges:
table = exploration.to_polars(
    "tracks", columns=["time_s", "speed"], start=0, stop=5_000
)

# Obtain a lazy Zarr handle without reading values when a library needs one:
array_handle = exploration.handle("tracks/speed")

# Produce a bounded, decimated Polars trace table for custom plotting:
trace = exploration.trace_frame(
    selected_path, column=11, start=0, stop=1_800, max_points=1_800
)
```

The helper defaults to at most 100,000 array elements per `read` and 10,000
rows per `to_polars` call. Narrow the selection instead of raising those limits
for routine interactive work.
                """
            ),
        ]
    )
    return


@app.cell
def _(analysis_dataset, exploration, selected_path):
    # Start here. Replace this expression or add cells below it; `exploration`
    # and the semantic dataset handle are defined in hidden cells and remain
    # available. `analysis_data` is None until a bounded copy is requested.
    # Do not output `analysis_data` here: a 100,000-row working copy would be
    # needlessly serialized into the notebook output. It remains globally usable.
    (exploration, analysis_dataset, selected_path)
    return


if __name__ == "__main__":
    app.run()
