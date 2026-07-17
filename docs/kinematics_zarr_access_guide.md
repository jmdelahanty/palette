# Accessing Kinematics Data in Palette Zarrs

This guide is for interns or new analysis users who need to read kinematics
results from an existing Palette analysis zarr. It assumes the analysis zarr
already exists and that track kinematics, swim bouts, and optionally bout/tail
kinematics have already been computed.

Use `scripts/py` from the Palette repo for all examples.

## Which Surface To Use

Palette stores related kinematics outputs in separate run families under
`analysis/`:

| Question | Preferred zarr surface | Preferred reader |
| --- | --- | --- |
| Frame-by-frame position, speed, heading, acceleration | `analysis/track_kinematics_runs/<scope>/<run>/` | `fisheye.analysis.track_kinematics_io.load_track_kinematics_track` |
| Swim-bout event boundaries and inter-bout intervals | `analysis/swim_bout_runs/<run>/` | `fisheye.analysis.swim_bout_io.load_swim_bout_tables` |
| Per-bout distance, duration, heading change, and eye-gaze metrics | `analysis/bout_kinematics_runs/<run>/` | `fisheye.analysis.bout_kinematics.resolve_bout_kinematics_tables` |
| Frame-by-frame tail-angle / tail-deflection traces | `analysis/tail_kinematics_runs/<run>/` | Direct zarr arrays for now |

Use the reader helpers when they exist. Several run families have historical
and compact layouts; the helpers hide those physical-layout differences.

Do not use `tracks/id_<track>/swim_bouts/` for new analyses. That is a legacy
compatibility mirror. Use `analysis/swim_bout_runs` and
`analysis/bout_kinematics_runs` instead.

## Discover Runs In A Zarr

```bash
scripts/py - <<'PY'
from pathlib import Path

from fisheye.shared.zarr_io import open_zarr_root

zarr_path = Path("/path/to/recording_analysis.zarr")
root = open_zarr_root(zarr_path, mode="r")
analysis = root["analysis"]

def show_parent(path):
    group = analysis
    for token in path.split("/"):
        if token not in group:
            print(f"{path}: missing")
            return
        group = group[token]
    names = sorted(str(name) for name in group.group_keys())
    print(f"{path}: latest={group.attrs.get('latest')!r}, runs={names}")

show_parent("track_kinematics_runs/offline")
show_parent("track_kinematics_runs/online")
show_parent("swim_bout_runs")
show_parent("bout_kinematics_runs")
show_parent("tail_kinematics_runs")
PY
```

Most downstream behavior analyses should start with the `offline` track
kinematics run unless there is a specific reason to use `online`.

## Load Frame-Level Track Kinematics

This loads one track from the latest offline track-kinematics run and converts
the most commonly used arrays into a Polars dataframe.

```bash
scripts/py - <<'PY'
from pathlib import Path

import polars as pl

from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.shared.zarr_io import open_zarr_root

zarr_path = Path("/path/to/recording_analysis.zarr")
track_id = 0

root = open_zarr_root(zarr_path, mode="r")
track = load_track_kinematics_track(
    root,
    run_name="latest",
    scope="offline",
    track_id=track_id,
)

positions_mm = track.positions_mm
if positions_mm is None:
    raise RuntimeError("This track-kinematics run has no positions_mm array.")

data = {
    "frame_index": track.frame_indices,
    "x_mm": positions_mm[:, 0],
    "y_mm": positions_mm[:, 1],
    "speed_filtered_mm_s": track.speed_mm_by_level.get("filtered"),
    "speed_smoothed_mm_s": track.speed_mm_by_level.get("smoothed"),
    "path_distance_filtered_mm": track.frame_path_distance_mm_by_level.get("filtered"),
    "heading_deg": track.heading_degrees,
    "heading_smoothed_deg": track.smoothed_heading_degrees,
    "delta_heading_deg": track.delta_heading_degrees,
    "delta_heading_smoothed_deg": track.delta_heading_smoothed_degrees,
    "sample_valid": track.sample_valid,
    "transition_valid": track.transition_valid,
}
if track.time_seconds is not None:
    data["time_s"] = track.time_seconds

df = pl.DataFrame({k: v for k, v in data.items() if v is not None})
print(track.run_path)
print(df.head())
print(df.select(pl.len().alias("rows")))
PY
```

Common speed levels are:

- `raw`: direct gap-aware speed before movement filtering.
- `filtered`: hysteresis-filtered speed; this is often the best default for
  behavioral summaries.
- `smoothed`: temporally smoothed speed.
- `averaged`: optional longer-window averaged speed.

The preferred physical units are `*_mm` and `*_mm_s`. Pixel arrays are present
for provenance and image-space visualization, but most behavioral analyses
should use millimetres.

## Load Swim-Bout Boundaries

Swim-bout runs define candidate bouts from one speed/detector signal. This is
where to read bout start/end frames, duration, inter-bout intervals, and the
segmentation parameters.

```bash
scripts/py - <<'PY'
from pathlib import Path

import polars as pl

from fisheye.analysis.swim_bout_io import load_swim_bout_tables
from fisheye.shared.zarr_io import open_zarr_root

zarr_path = Path("/path/to/recording_analysis.zarr")
root = open_zarr_root(zarr_path, mode="r")

payload = load_swim_bout_tables(
    root,
    run_name="latest",
    speed_level="filtered",
)

def records_to_frame(records):
    names = records.dtype.names or ()
    return pl.DataFrame({name: records[name] for name in names})

bouts = records_to_frame(payload.bouts)
intervals = records_to_frame(payload.inter_bout_intervals)

print(payload.level_path)
print("signal:", payload.signal.speed_level, payload.signal.signal_name)
print("bouts")
print(bouts.head())
print("inter-bout intervals")
print(intervals.head())
PY
```

Important bout-boundary fields include:

- `bout_id`
- `start_frame`, `end_frame`
- `core_start_frame`, `core_end_frame` when available
- `start_time_s`, `end_time_s`
- `duration_s`
- `path_length_mm`
- `net_displacement_mm`
- `mean_speed_mm_s`

Frame boundary fields are authoritative for slicing frame-level arrays.

## Load Per-Bout Kinematics Metrics

Bout-kinematics runs are the per-bout biological measurement layer. Use these
when you want bout duration, active path length, heading change, heading path,
and, by default, eye-gaze metrics aligned to bout rows. Historical or explicit
compatibility runs created with `--no-include-eye-gaze` may omit that level.

```bash
scripts/py - <<'PY'
from pathlib import Path

import polars as pl

from fisheye.analysis.bout_kinematics import resolve_bout_kinematics_tables
from fisheye.shared.zarr_io import open_zarr_root

zarr_path = Path("/path/to/recording_analysis.zarr")
root = open_zarr_root(zarr_path, mode="r")

parent = root["analysis"]["bout_kinematics_runs"]
run_name = parent.attrs.get("latest") or sorted(parent.group_keys())[-1]
run = parent[str(run_name)]

records_by_level, level_attrs, table_attrs = resolve_bout_kinematics_tables(run)

def records_to_frame(records):
    names = records.dtype.names or ()
    return pl.DataFrame({name: records[name] for name in names})

movement = records_to_frame(records_by_level["movement"])
heading = records_to_frame(records_by_level["heading_smoothed"])

print(f"analysis/bout_kinematics_runs/{run_name}")
print("source track run:", run.attrs.get("source_track_kinematics_run"))
print("source swim-bout run:", run.attrs.get("source_swim_bout_run"))
print("movement")
print(movement.head())
print("heading")
print(heading.head())
PY
```

Useful movement fields include:

- `physical_active_duration_s`
- `physical_active_path_length_mm`
- `physical_active_mean_speed_mm_s`
- `physical_active_peak_speed_mm_s`
- `physical_active_valid`

Useful heading fields include:

- `net_delta_heading_deg`
- `abs_net_delta_heading_deg`
- `within_heading_path_deg`
- `within_heading_range_deg`
- `within_heading_peak_to_peak_deg`
- `within_angular_velocity_mean_deg_s`
- `within_angular_speed_mean_deg_s`
- `within_window_valid`

If a field is invalid for a bout, check the corresponding `*_valid` field and
`failure_reason_bytes`.

## Direct Paths For Inspection

These are the most common physical paths when browsing a zarr manually:

```text
analysis/track_kinematics_runs/offline/<run>/tracks/id_<track>/frame_indices
analysis/track_kinematics_runs/offline/<run>/tracks/id_<track>/positions_mm
analysis/track_kinematics_runs/offline/<run>/tracks/id_<track>/movement/speed/filtered/mm
analysis/track_kinematics_runs/offline/<run>/tracks/id_<track>/movement/speed/filtered/frame_path_distance_mm
analysis/track_kinematics_runs/offline/<run>/tracks/id_<track>/heading_degrees
analysis/track_kinematics_runs/offline/<run>/tracks/id_<track>/smoothed_heading_degrees

analysis/swim_bout_runs/<run>/...
analysis/bout_kinematics_runs/<run>/...
analysis/tail_kinematics_runs/<run>/...
```

Prefer direct paths only for inspection or quick debugging. Analysis code
should use the reader helpers above.

## Alignment Rules

- Track arrays are ordered by `frame_indices`.
- Use `sample_valid` to identify valid per-frame samples when present.
- Use `transition_valid` for frame-to-frame quantities such as speed,
  acceleration, path distance, and angular velocity.
- Swim-bout rows define frame boundaries; use those boundaries to slice
  frame-level track arrays.
- Bout-kinematics rows are aligned to the source swim-bout rows and record the
  exact source run names in attrs.
- For GoodCopBadCop and other registered-coordinate analyses, use
  millimetre arrays when available and keep coordinate-frame provenance from
  the source run attrs.

## Minimal Sanity Checks

Before using a zarr for analysis, print:

- selected track-kinematics run path and `fps`
- available `track_ids`
- selected swim-bout run and speed level
- selected bout-kinematics run plus its `source_track_kinematics_run` and
  `source_swim_bout_run`
- row counts for the frame table, bout-boundary table, and per-bout metrics

If source run names do not match across layers, stop and resolve the mismatch
before computing summaries.

## Canonical References

- Full zarr layout: `src/fisheye/docs/zarr_structure.md`
- Track-kinematics reader: `src/fisheye/analysis/track_kinematics_io.py`
- Swim-bout reader: `src/fisheye/analysis/swim_bout_io.py`
- Bout-kinematics resolver: `src/fisheye/analysis/bout_kinematics.py`
- Bout-kinematics design: `docs/bout_kinematics_run_design.md`
