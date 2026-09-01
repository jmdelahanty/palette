# Cumulative path-distance composition implementation — 2026-09-01

## Decision

Cumulative distance traveled is a first-class recording measurement and a
composable cohort statistic. It is not computed independently inside each
plot.

The recording authority is the immutable provider-motion publication already
sealed by a validated recording-behavior bundle. The cohort authority is a
receipt-validated Phase-B export plus a selector-ineligible grouped-statistics
generation. Renderers are read-only views over those authorities.

No recording Zarr, provider-motion run, validated bundle, or Phase-B export
was recomputed for this extension.

## Recording-level surface

The general Palette explorer now exposes **Distance traveled** when one
complete physical unit level is present. It prefers millimetres and falls back
to pixels only when the complete millimetre surface is unavailable.

The projection consumes these already-persisted provider-motion arrays:

- `time_seconds` and `source_acquisition_frame_index`;
- `cumulative_path_distance_mm`;
- `frame_path_distance_smoothed_mm`;
- `delta_frames` and `delta_seconds`; and
- `transition_valid` and `transition_reason_code`.

The pixel equivalents are the exact fallback. A partial unit level is not
admitted.

The recording view contains:

1. the persisted cumulative smoothed-path trace;
2. explicit markers for invalid candidate transitions;
3. a per-second display aggregation made by summing valid persisted smoothed
   increments; and
4. valid-transition coverage for every displayed second.

The per-second table is a presentation projection, not a replacement
scientific authority. Its operation and input array names are recorded in the
projection metadata. A transition contributes only when it is a candidate,
the persisted transition-valid flag is true, the persisted increment is
finite, and its persisted elapsed time is positive. Invalid transitions are
excluded and counted; they are never converted into evidence of zero
movement.

Exact pre/training/post shading is available only through the semantic-epoch
source binding already sealed by the validated recording bundle. The adapter
requires exact closure between every `position_suite_epochs` interval and its
`semantic_role_bindings` record. It does not infer epochs from time, labels,
protocol conventions, or a nominal frame rate. An ordinary unvalidated core
source receives no guessed epoch shading.

The speed picker no longer includes cumulative- or frame-path arrays. Speed
and distance traveled are separate analysis identities with separate units
and display contracts.

## Cohort-level surface

The `distance_traveled` metric family contains four recording-weighted
metrics:

| Metric ID | Source | Recording reducer |
|---|---|---|
| `distance_traveled.session_total_path_mm` | `provider_motion_samples.cumulative_path_distance_mm` | exact terminal row |
| `distance_traveled.epoch_total_path_mm` | `epoch_behavior_summary.total_path_mm` | one exact epoch row |
| `distance_traveled.epoch_mean_speed_mm_s` | `epoch_behavior_summary.mean_speed_mm_s` | one exact epoch row |
| `distance_traveled.epoch_tracking_dropout_fraction` | `epoch_behavior_summary.tracking_dropout_fraction` | one exact epoch row |

The whole-session reducer is `terminal_at_max_order_v1`. Before selecting the
maximum `track_sample_row_id`, it requires within every
recording/condition/group unit:

- no null order or source-identity field;
- one constant bundle, provider, run, manifest, verification, and track
  identity;
- unique order values; and
- a gapless order interval.

It then requires exactly one maximum-order row. This selects the terminal value
of the already-persisted cumulative series; it does not sum thirteen million
export rows or reconstruct motion. Any identity divergence, duplicate order,
gap, or ambiguous terminal row fails closed.

The semantic-epoch metrics retain the existing exact-row reducer and declared
pre/training/post contrasts. Whole-session values use condition `__all__` and
remain visually separate from epoch values. Every cohort summary gives equal
weight to each finite recording, not each frame.

The static and interactive views consume one shared self-digested view payload.
The static report shows faint individual recording values, paired epoch lines,
median/IQR, mean, finite-recording counts, units, and statistics provenance.
The Marimo cohort explorer can select the same four metrics without reopening
the Phase-B sample table.

## Real-cohort evidence

Source Phase-B export manifest:

```text
230c30e032352c95bb9919f5704da6eba9d94a369b464089f575048639791d05
```

Distance-only canary:

```text
/tmp/goodbatbadbat-grouped-statistics-distance-canary-v002
record_sha256 = 50f744d3dd02987cdc071ec9f74847a2916aa79d077f921c7ae968cd32aa94d3
```

The canary contains 800 retained recording values, 10 descriptive rows, and
9 paired contrasts. All 80 complete recordings contribute finite values. The
four invalid semantic-protocol recordings remain parent-cohort
noncontributors.

Superseding full statistics generation:

```text
/tmp/goodbatbadbat-grouped-statistics-sandbox-v003
record_sha256 = af3ea7310138390663c4bf65b52f02ba6f1c7ccaa7575e81081edb264b477478
```

Superseding 13-figure report:

```text
/tmp/goodbatbadbat-grouped-statistics-report-v009
record_sha256 = 58633f5a8f001507756363396825d0d747877a1b4f1721a137eaf8039ffd14bd
```

The full generation contains 45 scalar metric specifications, two histogram
specifications, 15,191 retained recording values, 23,794 descriptive rows,
153 paired contrasts, 109,440 recording-histogram rows, and 1,368
histogram-cohort rows.

For the 80 contributing recordings, the whole-session path median is
5,209.19 mm (range 2,158.88–9,107.79 mm). These are exploratory cohort
descriptions, not acquisition-batch-adjusted or confirmatory results.

A real recording-bundle canary projected the first ten seconds into 1,001
rows, resolved all three exact semantic epochs, and produced eleven
second-index display rows. Two seconds had 98% valid-transition coverage,
demonstrating that the view preserves partial tracking coverage.

## Validation

- 62 focused tests pass outside the Codex sandbox; 15 predeclared explorer
  compatibility cases remain expected failures.
- The fail-closed reducer is covered for order gaps and mixed lineage.
- Semantic epoch interval/role divergence is covered.
- Static and interactive distance views round-trip through an immutable test
  statistics generation.
- The full real generation and all 13 static artifacts reopen through their
  strict readers.
- All 13 real interactive figures build and serialize from shared payloads.
- Both Marimo applications pass `marimo check` outside the sandbox.
- Python compilation, Black check, and `git diff --check` pass.

This is focused local evidence. Required repository CI remains mandatory before
merge, deployment, selector activation, or describing the branch as
merge-ready.

## Deferred extensions

- A long-form persisted per-second distance table could be added to a future
  export profile if repeated cross-recording time-series queries justify its
  storage. The current viewer aggregation is intentionally not treated as a
  new authority.
- Normalize cumulative distance to body lengths only after a reviewed,
  recording-bound body-length authority is defined.
- Add condition-duration-normalized distance comparisons only as explicit new
  metrics; do not compare raw epoch totals as if unequal epoch durations were
  equal.
- Consider a gap-aware confidence or sensitivity view after its scientific
  interpretation is specified. Tracking dropout is currently shown alongside
  distance rather than silently corrected.
