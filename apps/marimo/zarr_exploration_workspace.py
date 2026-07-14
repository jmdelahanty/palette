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

    from apps.marimo.components.zarr_workspace import ZarrExplorationWorkspace

    return Path, ZarrExplorationWorkspace, go, mo, np


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
                "This notebook is intentionally independent of Palette visualization "
                "contracts. Metadata browsing does not load array values, and preview "
                "reads enforce an element limit."
            ),
            mo.tree(zarr_workspace.summary(), label="Dataset summary"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    inventory_path = mo.ui.text(
        value="", label="Group path", placeholder="root (leave blank)"
    )
    inventory_depth = mo.ui.number(
        start=0, stop=8, step=1, value=2, label="Traversal depth"
    )
    inventory_limit = mo.ui.number(
        start=1, stop=2_000, step=25, value=250, label="Maximum nodes"
    )
    mo.hstack(
        [inventory_path, inventory_depth, inventory_limit],
        justify="start",
        gap=1,
        wrap=True,
    )
    return inventory_depth, inventory_limit, inventory_path


@app.cell(hide_code=True)
def _(inventory_depth, inventory_limit, inventory_path, mo, zarr_workspace):
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
def _(inventory_depth, inventory_limit, inventory_table, mo, zarr_workspace):
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
    else:
        inventory_selection_output = mo.md(
            "Select a dataset row above to reveal its metadata and contents."
        )
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
    selected_path = (
        str(contents_selected_rows[0]["path"])
        if contents_selected_rows
        else inventory_selected_path
    )
    selected_kind = inventory_selected_kind
    if selected_path:
        try:
            selected_info = zarr_workspace.info(selected_path)
            selected_kind = str(selected_info.get("kind", ""))
            selected_attrs = zarr_workspace.attrs(selected_path)
            active_source = (
                "group contents table"
                if contents_selected_rows
                else "main inventory table"
            )
            if selected_kind == "group":
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
                        f"### Active dataset: `{selected_path}`\n\n"
                        f"Selected from the {active_source}."
                    ),
                    active_guidance,
                    mo.hstack(
                        [
                            mo.tree(selected_info, label="Active metadata"),
                            mo.tree(selected_attrs, label="Active attributes"),
                        ],
                        justify="start",
                        gap=1,
                        widths="equal",
                    ),
                ]
            )
        except Exception as exc:
            selected_kind = ""
            active_selection_output = mo.callout(
                mo.md(f"Could not activate the selection: `{type(exc).__name__}: {exc}`"),
                kind="danger",
            )
    else:
        active_selection_output = mo.md("")
    active_selection_output
    return selected_kind, selected_path


@app.cell(hide_code=True)
def _(mo, selected_kind, selected_path):
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
    mo.vstack(
        [
            mo.md("## Array preview"),
            mo.md(
                f"Selected path: `{selected_path or 'none'}`. {preview_instruction}"
            ),
            mo.hstack(
                [preview_rows, preview_run],
                justify="start",
                gap=1,
                wrap=True,
            ),
        ]
    )
    return preview_rows, preview_run


@app.cell(hide_code=True)
def _(
    mo,
    preview_rows,
    preview_run,
    selected_kind,
    selected_path,
    zarr_workspace,
):
    if not selected_path:
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
def _(mo, np, selected_kind, selected_path, zarr_workspace):
    trace_supported = False
    trace_channel_lookup = {}
    trace_channel_picker = None
    trace_start = None
    trace_stop = None
    trace_max_points = None
    trace_run = None
    trace_coordinate_path = None
    if selected_kind == "array":
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
    if not trace_supported or trace_run is None:
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
exploration.channel_index(selected_path)       # named compact-dense columns

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
def _(exploration, selected_path):
    # Start here. Replace this expression or add cells below it; `exploration`
    # is defined in a hidden cell and remains available. Source reads are bounded.
    (exploration, selected_path)
    return


if __name__ == "__main__":
    app.run()
