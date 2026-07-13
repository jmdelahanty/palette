# Marimo Pair recording workspace

## Purpose

The recording workspace combines two views in one Marimo editor session:

1. the existing Palette recording explorer renders at the top with its
   implementation cells collapsed; and
2. an **Exploration workspace** at the bottom exposes editable cells over the
   live data already selected or loaded by the app.

It is an opt-in research surface. The existing `recording-app` remains the
stable, source-hidden `marimo run` service.

## Launch

Local Pixi launch:

```bash
pixi run -e recording recording-workspace -- \
  --zarr-path /path/to/recording_analysis.zarr
```

FileGlancer exposes the same task as **Recording Exploration Workspace**. It
requires exactly one **Recording Analysis Zarr** directory. Direct launches use
port 2721 by default; FileGlancer supplies its allocated port and access token.

The launcher makes a timestamped notebook copy under:

```text
~/.palette/marimo-workspaces/<session>/palette_recording_workspace.py
```

Set `PALETTE_RECORDING_WORKSPACE_ROOT` before a local launch to choose another
host location. The host path is printed once at startup. Inside the notebook it
is always `/workspace`.

## Filesystem authority

Marimo Pair can run arbitrary Python and can add, edit, delete, and execute
notebook cells. A Zarr object initially opened with `mode="r"` is therefore not
a sufficient protection: arbitrary code could otherwise reopen its host path
with a writable mode.

`run_recording_exploration_workspace.sh` uses Bubblewrap to create this mount
namespace:

| Sandbox path | Access | Contents |
| --- | --- | --- |
| `/data/recording.zarr` | read-only | Exactly the FileGlancer-selected recording |
| Palette checkout at its host path | read-only | Source plus the active Pixi environment |
| `/workspace` | read/write | Notebook copy, user analyses, and explicit outputs |
| `/tmp` and cache paths | read/write | Private directories inside the session workspace |
| minimal `/usr`, `/etc`, `/sys`, `/proc`, `/dev` | read-only or synthetic | Python and operating-system runtime support |

The user's home directory, registry, other recordings, and other shared
filesystem roots are not mounted. The launcher rejects collection and registry
arguments. It also clears the inherited process environment before adding only
the values required by the notebook runtime.

Bubblewrap and unprivileged user namespaces are mandatory. Failure to create
the namespace is a launch failure; there is no unsandboxed fallback.

## Exploration handle

The visible starter cell defines `exploration`. It follows the recording,
provider, run, and analysis selected in the controls above. Its compact public
surface includes:

```python
exploration.summary()
exploration.core_frame             # selected Polars LazyFrame, when available
exploration.related_core_frames    # related LazyFrames, such as bout events
exploration.chaser_tables          # viewer-native loaded Pandas/Polars tables
exploration.open_zarr()            # read-only Zarr group
```

The object does not load additional dense arrays merely for display. Existing
bounded/deferred Zarr projections and Polars lazy frames retain their normal
semantics. Users may add ordinary Marimo cells after the starter cell.

## Connecting Marimo Pair

[Marimo Pair](https://marimo.io/blog/marimo-pair) is installed in the user's coding-agent environment, not in the
Palette Pixi environment. Current upstream prerequisites are `bash`, `curl`,
and `jq`. Installation is intentionally not performed by Palette; see the
[upstream skill repository](https://github.com/marimo-team/marimo-pair):

```bash
npx skills add marimo-team/marimo-pair
```

After FileGlancer opens the editor and establishes an active notebook session:

1. give the agent the compute-node Marimo URL;
2. set `MARIMO_TOKEN` in the agent process to the same access token contained
   in FileGlancer's authenticated service URL; and
3. ask the agent to pair with that URL and append analyses below the exploration
   starter cell.

Do not paste the access token into a notebook cell, chat prompt, saved script,
or job log. The Pair execution helper reads it from `MARIMO_TOKEN` and sends it
as a bearer token.

Pair's code-mode API is private and unversioned. Palette pins Marimo 0.23.3,
which includes code mode, but compatibility with the installed Pair skill must
be tested when either side changes.

## Deliberate restrictions

- Pair may edit and save the per-session notebook copy, not the canonical
  Palette notebook.
- Dataset and Palette mounts remain read-only even if notebook code requests a
  writable Zarr mode or attempts a direct file write.
- Package/environment mutation is not supported in the first canary because
  the Pixi environment is read-only. Add dependencies through reviewed Palette
  packaging changes.
- The first workspace exposes one recording only. Registry and sibling-recording
  browsing can be reconsidered after a multi-mount authorization contract exists.
- The workspace does not write analysis results back into the registry or
  exported dataset. Explicit user outputs belong beneath `/workspace`.
