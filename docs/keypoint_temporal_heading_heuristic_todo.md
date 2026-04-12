# Keypoint Temporal Heading Heuristic TODO

<!-- design-meta
status: draft
last_updated: 2026-04-06
-->

## Purpose

Define a practical review heuristic for catching keypoint rows whose heading is
temporally implausible relative to nearby frames.

The motivating failure mode is:

- keypoints flip orientation between adjacent frames
- heading changes by a very large amount even though the fish motion is smooth
- the row may still be `heading_usable=true`, so current gating does not catch
  it

This note is intentionally about:

- review prioritization
- quality diagnostics
- future optional training gates

It is not part of the current staleness/detect-crop pass.

Related notes:

- [keypoint_heading_validity_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_heading_validity_todo.md)
- [keypoint_quality_registry_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_quality_registry_workflow.md)
- [preferred_detect_crop_runs_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/preferred_detect_crop_runs_design.md)

## Short Answer

Yes, large frame-to-frame heading deviation is a good review heuristic.

But it should start as:

- a heuristic
- circular-angle aware
- gated on usable neighboring rows
- review/QC metadata only

It should not start as:

- an automatic hard failure
- a silent rewrite
- a training exclusion without validation

## Why This Is Useful

Current heading fields answer:

- is the heading finite?
- is the heading usable for downstream consumers?

They do not answer:

- is this heading plausible relative to nearby frames?

That leaves a real blind spot for cases like:

- left/right eye swap
- swim-bladder / eye ordering flip
- skeleton inversion with otherwise finite geometry
- isolated bad rows inside an otherwise clean refined run

## Core Design Rule

Temporal heading consistency must use circular angle differences.

Do not use naive subtraction.

Correct behavior:

- `179° -> -179°` is a `2°` change
- not a `358°` change

Recommended primitive:

- `delta = abs(wrap_to_180(curr_heading - prev_heading))`

where `wrap_to_180` maps into `[-180, 180)`.

## Recommended First-Pass Heuristic

For each row, compute:

- `heading_delta_prev_deg`
- `heading_delta_next_deg`
- `heading_temporal_outlier`
- optional `heading_flip_suspect`

Only evaluate when:

- current row has `heading_usable=true`
- neighboring row has `heading_usable=true`
- neighboring row is close enough in time
- neighboring row belongs to the same logical identity when identity exists

Recommended first-pass window:

- immediate previous/next usable row
- within a configurable frame-gap threshold

## Recommended Thresholding Approach

Do not start with one global hard threshold that excludes data.

Start with a review heuristic such as:

- flag if `min(delta_prev, delta_next) > threshold_deg`
- or flag if both neighbor deltas are large
- or flag if delta is large and translational motion is small

Good default exploration thresholds might be:

- `90°`
- `120°`
- `150°`

but they should be validated on real reviewed data before becoming any form of
gate.

## Identity Caveat

This heuristic is only as good as the row identity it uses.

Near-term:

- for current single-subject-per-arena training data, adjacent rows within the
  same recording may be good enough for an initial heuristic

Long-term:

- this should run on stable `(frame, entity)` identity rather than raw sparse
  ROI row order
- that aligns naturally with the current refined-detect direction:
  canonical curated detect reads from `refined_detect_runs/<run>/instances`,
  with denser track- or slot-shaped projections treated as derived outputs

So this heuristic should not assume that "neighboring sparse rows" always means
"same fish."

## Suggested Output Locations

Possible homes for the first pass:

### Option A: refined keypoint run attrs / summary only

Store only aggregate counts such as:

- `heading_temporal_outlier_count`
- `heading_temporal_outlier_rate`

Pros:

- cheap
- simple

Cons:

- not enough for targeted review

### Option B: per-row arrays in `refined_keypoints_runs/<run>`

Example arrays:

- `heading_delta_prev_deg`
- `heading_delta_next_deg`
- `heading_temporal_outlier`

Pros:

- supports targeted review and diagnostics
- easiest to inspect later

Cons:

- expands run schema

### Option C: registry/profile projection only

Project only summary counts/rates to quality tables.

Pros:

- useful for preflight

Cons:

- loses row targeting unless arrays exist on disk first

Recommended path:

- write per-row arrays on refined keypoint runs
- project summary counts/rates into registry quality views

Policy refinement:

- disable temporal-heading review for sampled imports and other archives where
  temporal continuity is not meaningful
- record that choice explicitly in run summary metadata rather than silently
  treating the archive as `0` outliers
- keep sampled-import training zarrs out of temporal-outlier review queues

## Initial Policy Recommendation

Phase 1:

- compute the heuristic
- expose it in review/diagnostics
- do not change training gates

Phase 2:

- study false positives on reviewed runs
- tune thresholds and neighbor-gap behavior

Phase 3:

- optionally add a soft quality gate or warning threshold in training prep

## Non-Goals

- do not auto-correct headings
- do not silently flip keypoints based on temporal smoothing alone
- do not make this a hard failure before validation
- do not conflate temporal outlier status with `heading_usable`

## Concrete Next Steps

1. Add a small design/contract for per-row temporal heading arrays on refined
   keypoint runs.
2. Implement circular heading-delta computation on a refined run.
3. Surface flagged rows in keypoint review or a quality diagnostic export.
4. Project summary counts/rates into keypoint quality registry rows.
5. Revisit whether any threshold should influence training selection.
