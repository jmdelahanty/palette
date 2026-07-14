# Marimo Pair source-Zarr workspace

## Purpose

The **Palette Zarr Exploration Workspace** opens any one selected Zarr in an
editable Marimo session. Unlike the Recording Explorer and Recording
Exploration Workspace, it does not require persisted Palette visualization
specs and does not assume the selected directory is an analysis Zarr.

This makes it the appropriate FileGlancer app for source, training, analysis,
or other Zarr layouts when the goal is open-ended inspection with a person or
Marimo Pair agent.

## Launch

Local Pixi launch:

```bash
pixi run -e recording zarr-workspace -- \
  --zarr-path /path/to/source.zarr
```

FileGlancer discovers
`apps/fileglancer/zarr_workspace/runnables.yaml` as an independent app. Its
required **Source Zarr** parameter renders FileGlancer's directory Browse
selector. The user chooses the dataset before launch, and only that directory
is mounted inside the session at `/data/source.zarr`. The current FileGlancer
manifest contract has no separate Zarr-only filter field, so the picker can
browse directories generally; the notebook validates that the chosen directory
opens as a Zarr.

The launcher makes a timestamped notebook copy under:

```text
~/.palette/marimo-zarr-workspaces/<session>/palette_zarr_workspace.py
```

Set `PALETTE_ZARR_WORKSPACE_ROOT` before a local launch to choose a different
host location. The host notebook path is printed once at startup.

## Filesystem authority

`scripts/run_zarr_exploration_workspace.sh` uses the same fail-closed
Bubblewrap boundary as the analysis-aware recording workspace:

| Sandbox path | Access | Contents |
| --- | --- | --- |
| `/data/source.zarr` | read-only | Exactly the FileGlancer-selected Zarr |
| Palette checkout at its host path | read-only | Source plus the active Pixi environment |
| `/workspace` | read/write | Notebook copy and explicit derived outputs |
| private `/tmp` and cache paths | read/write | Session-local runtime files |
| minimal runtime paths | read-only or synthetic | Python and OS support |

No registry, sibling recording, home directory, or shared filesystem root is
mounted. The selected source remains read-only even though Marimo Pair can run
arbitrary Python. There is no unsandboxed fallback when Bubblewrap or user
namespaces are unavailable.

## Stable exploration interface

A hidden initialization cell assigns a `ZarrExplorationWorkspace` to the stable
name `exploration`. The visible bottom cell only consumes that handle, so a
person or Pair agent may replace or delete the starter expression without
removing the interface. The helper is deliberately independent of Palette
schemas:

```python
exploration.summary()
exploration.ls("", max_items=100)
exploration.walk("tracks", max_depth=2, max_items=250)
exploration.find("speed")
exploration.info("tracks/speed")
exploration.attrs("tracks")
exploration.handle("tracks/speed")
```

The metadata inventory uses a bounded master-detail selection pattern.
Selecting an array reveals its metadata and makes it the active `selected_path`.
Selecting a group reveals its metadata plus a second selectable table of its
bounded descendants. Selecting an array from that contents table makes the
descendant active and enables preview; selecting a nested group keeps it as a
metadata selection while its array descendants remain available in the same
table. Both tables use the inventory traversal-depth and node-limit controls,
so opening a group does not trigger unbounded metadata discovery or array
reads. The active path is available to editable cells and Pair agents as
`selected_path`.

These calls inspect handles and metadata. They do not materialize array values.
Node paths are relative to the selected Zarr and reject absolute paths or
`..` traversal.

Array reads accept only integer, slice, and ellipsis indexing and default to a
100,000-element limit:

```python
speed = exploration.read("tracks/speed", slice(0, 1_000))
image_crop = exploration.read(
    "images/frames", (0, slice(100, 200), slice(100, 200))
)
```

### Numeric trace plotting

Selecting a numeric one- or two-dimensional array reveals a **Numeric trace
plot** panel. For known compact-dense Palette arrays such as `frame_angles`, the
workspace resolves human-readable channel names through the persisted channel
index. It also uses a matching time coordinate such as
`support/frame_time_seconds` when available. The eye-angle defaults are the
smoothed left, right, and vergence channels.

The initial plot window is 1,800 rows, and no interactive plot may scan more
than 100,000 source rows. Up to 5,000 points are displayed by default using a
line-only WebGL trace. This distinction matters for matrices whose physical
chunks contain every channel: display decimation reduces browser work but does
not avoid reading every physical chunk intersecting the requested source
window.

Pair agents and custom cells can use the same bounded path without Plotly:

```python
channels = exploration.channel_index(selected_path)
trace = exploration.trace_frame(
    selected_path,
    column=11,
    start=0,
    stop=1_800,
    max_points=1_800,
)
```

`trace` is a Polars DataFrame containing `row_index`, `value`, and
`time_seconds` when a matching coordinate is available.

Sibling one-dimensional arrays can be loaded directly into a bounded Polars
DataFrame, without Pandas or a PyArrow conversion bridge:

```python
frame = exploration.to_polars(
    "tracks",
    columns=["time_s", "speed"],
    start=0,
    stop=5_000,
)
```

The default table limit is 10,000 rows. These are interaction safety defaults,
not scientific downsampling contracts. Narrow the source selection for normal
exploration. If a larger computation is scientifically necessary, write a
reviewed batch workflow rather than materializing a million-frame array in the
browser process.

## Pair-agent handoff

The notebook itself contains the same API examples immediately above the
visible starter cell. A Pair prompt can therefore be concise:

> Inspect `exploration.summary()` and a bounded metadata inventory first. Use
> explicit slices for all array reads, and save any derived outputs only under
> `/workspace`.

Connect Pair to the authenticated Marimo editor using the FileGlancer service
URL and put its access token in the agent process's `MARIMO_TOKEN` environment
variable. Do not put access tokens in notebook cells, prompts, scripts, or job
logs. Pair installation and version caveats are shared with
[the analysis-aware recording workspace](marimo_pair_recording_workspace.md#connecting-marimo-pair).

## Choosing among recording apps

- **Palette Recording Explorer**: locked, rendered views backed by persisted
  visualization contracts.
- **Palette Recording Exploration Workspace**: those rendered views plus an
  editable cell over the currently selected analysis.
- **Palette Zarr Exploration Workspace**: generic editable inspection of any
  single Zarr, with no visualization-contract requirement.

All three keep their selected datasets read-only. Only the two exploration
workspaces expose writable per-session notebook directories.
