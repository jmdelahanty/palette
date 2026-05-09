# Crimson Swim-Bout Compact-V2 Read Contract

## Status

Design handoff for Crimson. Palette can now write compact swim-bout runs behind
`detect_bouts_multi_level --layout compact_v2`, but the default writer remains
hierarchical v1 until Crimson can read the compact layout directly.

Crimson implementation update, 2026-05-09: the Crimson agent reported a focused
compact swim-bout GUI smoke script, `scripts/gui_smoke_compact_swim_bouts.sh`,
and successful workstation-display smoke runs on all three compact-v2 audit
archives:

```text
arena 2 Feeding:
  /tmp/crimson_compact_swim_bout_smoke_arena2_20260509.log
  [SwimBouts] Loaded compact-v2 run '...compact_v2_canary' candidate 0
  [SwimBouts] Loaded 20 candidates
  Successfully loaded zarr file

arena 1 DefaultScreen:
  /tmp/crimson_compact_swim_bout_smoke_defaultscreen_20260509.log
  [SwimBouts] Loaded compact-v2 run '...compact_v2_audit_20260509' candidate 0
  [SwimBouts] Loaded 15 candidates
  Successfully loaded zarr file

arena 3 Feeding:
  /tmp/crimson_compact_swim_bout_smoke_arena3_20260509.log
  [SwimBouts] Loaded compact-v2 run '...compact_v2_audit_20260509' candidate 0
  [SwimBouts] Loaded 10 candidates
  Successfully loaded zarr file
```

That confirms the compact-v2 Crimson loader branch is reached during a real
`redgui` startup on each audit archive. Candidate counts differ by archive
because Crimson filters by compatible track/source inventory; the count is not a
fixed schema-level expectation. A fresh compact-v2-only promoted canary remains
the final validation gate before Palette flips its default writer.

Palette validation on 2026-05-09 showed logical equivalence between the current
hierarchical run and compact v2 through `fisheye.analysis.swim_bout_io`:

- feeding canary: 68/68 checks passed, 519 default-signal bouts, max numeric
  drift 0
- moving-grating/default-screen canary: 68/68 checks passed, max numeric drift
  0
- second feeding archive: 68/68 checks passed, max numeric drift 0
- object-count reduction for each matching run: 493 v1 `zarr.json` files to
  145 compact-v2 `zarr.json` files

## Original Crimson Gap

Read-only audit of
`/home/delahantyj@hhmi.org/gitrepos/crimson-ui-monolith/src/zarr_loader_movement.cpp`
found that Crimson currently discovers swim-bout candidates by looking for:

```text
analysis/swim_bout_runs/<run>/<speed_level>/bouts/start_frame
analysis/swim_bout_runs/<run>/<speed_level>/bouts/end_frame
```

That is the hierarchical v1 shape only. A compact-v2 run stores bout rows at
`analysis/swim_bout_runs/<run>/tables/bouts` with `candidate_id` and `signal_id`
columns, so it will be invisible to Crimson until a compact branch is added.

## Required Reader Behavior

Crimson should treat `analysis/swim_bout_runs` as a logical candidate surface,
not as a fixed `<run>/<speed_level>` path.

For each run under `analysis/swim_bout_runs/<run>`:

1. Read run attrs.
2. If `attrs["layout"] == "compact_tabular_v2"` or `indexes/candidates` exists,
   use the compact-v2 branch.
3. Otherwise, keep the existing hierarchical-v1 branch.

## Compact-V2 Branch

Read:

```text
indexes/candidates
indexes/signal_variants
tables/bouts
tables/peak_events                  # optional for timeline metadata
tables/inter_bout_intervals          # optional
signals/detector_signal_mm_s         # optional dense detector traces
signals/detector_signal_signal_ids   # maps dense signal rows to signal_id
signals/frame_indices                # frame axis for dense detector traces
```

Candidate selection:

- Use `run.attrs["default_candidate_id"]` when present.
- Otherwise choose the candidate row with `is_default == true`.
- Otherwise choose the first candidate row.

Signal selection/discovery:

- Create one Crimson `SwimBoutSeries` per `indexes/signal_variants` row for
  the selected candidate.
- Preserve `speed_level` from the signal row, for example
  `speed_exponential`.
- Preserve `role`, especially `detector_response` versus
  `physical_estimator`.
- Preserve `source_level`, `path_distance_source_level`, `tau_s`, and
  `transform_type` for compatibility matching and UI labels.
- Set `is_default_level` when `signal_id == run.attrs["default_signal_id"]`.

Bout table filtering:

- Load `tables/bouts`.
- Filter rows where `candidate_id == selected_candidate_id` and
  `signal_id == selected_signal_id`.
- Populate the existing Crimson fields from the filtered rows:
  `start_frame`, `end_frame`, `core_start_frame`, `core_end_frame`,
  `start_time_s`, `end_time_s`, `duration_s`, `path_length_mm`,
  `path_length_px`, `net_displacement_mm`, `net_displacement_px`,
  `peak_detection_signal_mm_s`, `peak_physical_speed_mm_s`, and
  `gap_censored` when present.

Detector trace:

- If `signals/detector_signal_mm_s` exists, read
  `signals/detector_signal_signal_ids`.
- Find the row whose value equals the selected `signal_id`.
- Use that row as the detector trace and read `signals/frame_indices` as the
  frame axis.
- This replaces the v1 path
  `<run>/<speed_level>/detection_signal_mm_s`.

## Compatibility Rules

Crimson's existing compatibility logic should continue to work if it compares
logical fields:

- Track source: `source_track_kinematics_run`
- Track identity: `track_id`
- Selected speed: direct match to `speed_level`, or source match against
  `source_level` / `path_distance_source_level`

For compact detector signals, `speed_level == "speed_exponential"` should be
compatible with a selected `filtered` physical speed when
`source_level == "speed_filtered"` or
`path_distance_source_level == "filtered"`.

## UI Labels

Suggested compact-aware label fields:

```text
<run_name> | <speed_level without speed_> | <n_bouts> bouts |
role <physical_estimator|detector_response> |
candidate <candidate_id> signal <signal_id> |
default/latest flags
```

The UI should not expose compact physical paths as if they were stable API.
Use run name, candidate ID, signal ID, and speed level as the stable selection
identity.

## Acceptance Checks

Use these Palette archives for validation:

```text
/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr
/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr
/nvme1/recordings/2026-01-28T23-15-10Z_arena_3_Feeding/zarr/2026-01-28T23-15-10Z_arena_3_Feeding_analysis.zarr
```

Expected result:

- hierarchical v1 candidates still load
- compact-v2 audit candidates load when selected by run name
- default compact signal is `speed_exponential`
- compact candidate counts match the compatible track/source inventory for the
  selected archive
- detector trace appears for `speed_exponential`
- timeline rectangles and core rectangles render from compact rows
- selecting a filtered track speed still offers the compatible exponential
  detector-response candidate
