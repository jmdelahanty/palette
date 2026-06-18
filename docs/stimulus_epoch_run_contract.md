# Stimulus Epoch Run Contract
<!-- contract-meta
status: draft
last_updated: 2026-06-17
-->

Purpose: define the per-recording analysis surface for reusable event-aligned
time windows. This keeps stimulus-window semantics in one place so detection,
pose, mask, tracking, and stimulus-response analyses do not each invent their
own pre/training/post or trial-window definitions.

## Boundary

`analysis/stimulus_runs/<run>` remains the imported stimulus/event authority.
It owns raw event timing, protocol metadata, calibration, and alignment
metadata.

`analysis/stimulus_epoch_runs/<run>` is a derived analysis run. It resolves
named event windows from one exact stimulus run and records the frame intervals
that downstream analyzers should consume.

Downstream modality-specific analyses should measure signals inside these
windows, not define the windows themselves. Examples:

- `analysis/detection_occupancy_runs/<run>` measures refined-detection coverage
  and spatial occupancy per epoch.
- `analysis/chaser_distance_runs/<run>` measures offline fish-to-chaser
  distance per frame and summarizes those distances per epoch.
- future keypoint, mask, track, or bout analyses may measure their own
  per-epoch summaries against the same epoch run.
- `analysis/stimulus_response_runs/<run>` may consume epoch windows when its
  response windows match a reusable protocol-level definition, while still
  owning higher-level biological response metrics.

## Storage

Recommended canonical location:

```text
analysis/stimulus_epoch_runs/<run>/
  zarr.json
  windows/
    window_id
    label_bytes
    start_frame
    end_frame
    start_time_s
    end_time_s
    duration_s
    source_start_event_name_bytes
    source_end_event_name_bytes
    source_start_event_frame
    source_end_event_frame
    source_policy_bytes
```

The parent group should carry:

```text
analysis/stimulus_epoch_runs.attrs["latest"] = <run>
```

## Required Run Attributes

- `schema_id`: stable schema name, for example
  `"palette.stimulus_epoch_windows.v1"`.
- `schema_version`: integer schema version.
- `method`: resolver method, for example `"goodcopbadcop_chaser_epochs"`.
- `method_version`: implementation/contract version.
- `row_axis`: `"epoch_windows"`.
- `source_stimulus_run`: exact source run name.
- `source_stimulus_path`: exact source path, for example
  `analysis/stimulus_runs/stimulus_external_ipc_20260616_01`.
- `source_event_schema`: event-name/frame columns used to resolve windows.
- `epoch_policy`: protocol-specific policy name and parameters.
- `source_refs`: standard source refs payload.
- `parameters`: user-visible resolver parameters.
- `created_at_utc`: creation timestamp.
- `provenance`: standard Palette provenance payload when available.
- lineage fingerprint attrs when implemented.

## GoodCopBadCop Epoch Policy

The current GoodCopBadCop heatmap workflow resolves three event-aligned windows
from chaser/protocol events:

| Label | Start event | End event |
| --- | --- | --- |
| `pre_event` | `CHASER_PRE_PERIOD_START` or `PROTOCOL_START` fallback | frame before `CHASER_TRAINING_START` |
| `training_event` | `CHASER_TRAINING_START` | frame before `CHASER_POST_PERIOD_START` |
| `post_event` | `CHASER_POST_PERIOD_START` | frame before `PROTOCOL_FINISH`, `PROTOCOL_STOP`, `STEP_END`, or `CHASER_PRESENTATION_END` fallback |

If the pre-start event is missing, the resolver may use frame `0` and record
that fallback in `source_policy_bytes` or run attrs. If the finish event is
missing, the resolver may use the recording's final frame and record that
fallback.

The canonical writer lives in `fisheye.analysis.stimulus_epoch_runs`. The older
plotting resolver in
`fisheye.visualization.plot_detection_epoch_heatmaps.resolve_stimulus_event_windows`
is the prototype implementation that informed this run family, but new
downstream code should consume `analysis/stimulus_epoch_runs/<run>` instead of
re-resolving event windows locally.

For the current GoodCopBadCop writer/review workflow that consumes these epoch
windows, see
[`goodcopbadcop_analysis_surfaces.md`](goodcopbadcop_analysis_surfaces.md).

## Consumer Rule

Consumers must store the exact epoch source they used:

```text
source_stimulus_epoch_run = <run>
source_stimulus_epoch_path = analysis/stimulus_epoch_runs/<run>
source_stimulus_epoch_schema_id = palette.stimulus_epoch_windows.v1
```

Consumers may cache copied window labels and frame bounds for convenience, but
those copies are not the authority. If the epoch policy is corrected, downstream
runs should be regenerated or marked stale rather than patched in place.

## Why Not Store These In Detection Occupancy?

Detection occupancy, keypoint summaries, mask summaries, and tracking summaries
can all be event-aligned. If each analysis family defines its own windows, the
same words such as "training" or "post" can silently mean different frame
intervals. A dedicated `stimulus_epoch_runs` surface makes window identity
reusable and auditable before modality-specific measurements are computed.
