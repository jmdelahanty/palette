# Interrupted legacy stimulus sessions (flag record, 2026-09-23)

_Supporting evidence for design decision D5 in [README.md](README.md)._

A read-only census of the latest stimulus run in every analysis Zarr under the
recordings store found **4 of 239** runs that stopped without finishing. They
are the four arenas of one acquisition. Nothing was modified.

| Recording | Latest stimulus run |
|---|---|
| `2026-07-02T15-06-50Z_arena_1_GoodCopBadCop` | `stimulus_20260702_164032` |
| `2026-07-02T15-06-50Z_arena_2_GoodCopBadCop` | `stimulus_20260702_164105` |
| `2026-07-02T15-06-50Z_arena_3_GoodCopBadCop` | `stimulus_20260702_164138` |
| `2026-07-02T15-06-50Z_arena_4_GoodCopBadCop` | `stimulus_20260702_164212` |

## What happened

Each run has `PROTOCOL_START` and `PROTOCOL_STOP` but no `PROTOCOL_FINISH`,
and one `STEP_START` with no `STEP_END`. Event times for arena 1
(session clock): pre-period 1.3–601.3 s, training 601.3–781.3 s, post-period
starts 781.3 s, `PROTOCOL_STOP` at 849.9 s. **The post-period ran about 69 s
of a 600 s recipe post-period** (recipe step `duration_s = 1380`).

The legacy importer recorded the single CHASER step as `start_camera_frame=741`,
`end_camera_frame=742` (one frame) while keeping the recipe `duration_s = 1380`.

## Exposure

- These analysis Zarrs contain no chaser analysis components
  (`analysis/` holds only `stimulus_runs`, `acquisition_video_streams`, `enums`,
  `calibration`), so no Palette chaser result was computed from them as of
  2026-09-23.
- Any result produced elsewhere that includes them treats a 69 s post-period as
  a full one: legacy chaser window resolution ends the post-period at
  `PROTOCOL_STOP`. Check cohort membership before reusing such results.

## Handling until D5 lands

Exclude these four runs from any analysis that needs the post-period, and record
the reason. They remain valid for the pre-period and training windows, which
completed. The D5 step-requirement check must detect this case generically:
`PROTOCOL_STOP` without `PROTOCOL_FINISH`, and a CHASER step whose post phase is
shorter than its recipe, not only a missing step.

## Census method and coverage

For each `recordings/*/zarr/*_analysis.zarr`, the latest stimulus run's
`events/event_name` was decoded and counted; a run is interrupted when it has
`PROTOCOL_STOP` without `PROTOCOL_FINISH`, or fewer `STEP_END` than
`STEP_START`. 19 Blindfish runs store `event_type_id` without event names and
could not be classified by name; for all 23 Blindfish runs every recipe step is
materialized with normal frame spans, so there is no sign of interruption.
