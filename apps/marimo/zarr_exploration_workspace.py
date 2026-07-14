#!/usr/bin/env python3
"""Editable, read-only-mounted workspace for an arbitrary source Zarr."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo

    from apps.marimo.components.zarr_workspace import ZarrExplorationWorkspace

    return Path, ZarrExplorationWorkspace, mo


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
