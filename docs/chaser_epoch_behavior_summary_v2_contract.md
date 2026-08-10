# Chaser Epoch Behavior Summary v2 Contract

Date: 2026-08-10

Status: implementation checkpoint; production activation still requires the
analytics-export and visualization consumers to adopt v2.

## Purpose

`palette.chaser.epoch_behavior_summary.v2` removes three ambiguity classes from
the maintained epoch summary:

- no implicit physical speed level;
- no speed or path samples across invalid track transitions;
- no behavioral event rate divided by unobserved wall-clock time.

The immutable default child is `kinematics_bouts_v2`. It is deliberately distinct
from the historical `kinematics_bouts_v1` child, so recomputation cannot overwrite
or silently reinterpret an existing v1 publication.

## Required authorities

The `authoritative_v2` builder must resolve all of the following or fail before
staging:

1. a positive, finite `fps` on the selected chaser-distance run;
2. an explicit physical track speed level from `raw`, `filtered`, `smoothed`, or
   `averaged`;
3. a verified canonical offline track-kinematics run containing the selected
   millimetre speed and frame-path arrays;
4. row-aligned persisted `sample_valid` and `transition_valid` arrays;
5. a verified swim-bout run bound to the same track-kinematics run.

Missing authorities are terminal. The authoritative writer also rechecks the
result's schema/method identity, exact source bindings, explicit speed-selection
receipt, positive FPS, and denominator columns before sealed atomic publication.
It cannot publish a `status=complete` component carrying a missing speed or bout
warning.

The multi-recording runner performs the speed-level check before discovering or
opening any target. Its default path is authoritative v2; callers must pass
`--speed-level <level>`. The only route to the legacy builder and legacy sealed
writer is the separate `--legacy-v1-compatibility` flag.

## Validity and denominator rules

For an inclusive epoch `[start_frame, end_frame]`:

- `valid_tracked_frame_count` counts track rows inside the epoch where
  `sample_valid` is true;
- `valid_tracked_duration_s = valid_tracked_frame_count / fps`;
- speed and path samples require `sample_valid & transition_valid`;
- `bout_rate_per_min` and `inter_bout_interval_rate_per_min` divide their counts by
  `valid_tracked_duration_s / 60`, and are `NaN` when that duration is zero;
- `wall_fraction` divides `wall_frame_count` by finite, in-arena, valid fish-center
  samples.

These rules are persisted in both table columns and component parameters. The v2
`per_epoch_fish` table adds:

| Field | Meaning |
| --- | --- |
| `valid_tracked_frame_count` | persisted valid track rows in the epoch |
| `valid_tracked_duration_s` | valid row count divided by declared FPS |
| `valid_tracked_duration_source` | exact duration construction rule |
| `motion_valid_sample_count` | rows satisfying both maintained validity arrays |
| `motion_validity_rule` | `sample_valid_and_transition_valid` |
| `wall_fraction_denominator_count` | denominator used by `wall_fraction` |
| `wall_fraction_denominator` | `valid_in_arena_center_samples` |
| `bout_rate_denominator_s` | seconds used by `bout_rate_per_min` |
| `bout_rate_denominator` | `valid_tracked_duration_s` |
| `inter_bout_interval_rate_denominator_s` | seconds used by the interval rate |
| `inter_bout_interval_rate_denominator` | `valid_tracked_duration_s` |

## Legacy compatibility boundary

Historical source-tolerance behavior is available only by explicitly selecting
`legacy_v1_compatibility` and publishing with
`write_legacy_chaser_epoch_behavior_summary_component`. That mode retains schema
`palette.chaser.epoch_behavior_summary.v1`, method version 1, wall-clock event-rate
denominators, and the historical speed/source fallbacks. Invalid or absent FPS remains
terminal in both modes. The CLI exposes compatibility only through
`--legacy-v1-compatibility`.

The maintained v2 writer rejects a v1 result, and the legacy writer rejects a v2
result. Compatibility output is not evidence for v2 scientific correctness.

## Integration dependency

The maintained chaser analysis profiles and derived-surface catalog declare v2 in
this checkpoint. Before production activation, the visualization loader and
cross-recording analytics export must add explicit v2 support and retain v1 only as
a version-dispatched compatibility schema. The v2 export contract must carry the
denominator receipt fields rather than dropping them.
