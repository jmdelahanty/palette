# Per-axis channel lists for compact eye-angle runs

- **Status:** stage 1 implemented on `agent/palette/eye-angle-per-axis-channels-20261007`; stages 2-4 are a proposal.
- **Owner:** Jeremy Delahanty.
- **Last reviewed:** 2026-10-07.
- **Builds on:** [compact-dense-v2 design](../../eye_angle_compact_v2_design.md), [physical column layout](../../eye_angle_physical_layout.md).
- **Why now:** all 84 goodbatbadbat v3 eye-angle runs are selector-ineligible candidates, and their gaze convention review (`operations/eye_gaze_convention_review_goodbatbadbat84_6fda0822_20260831_v1`) still has 84/84 decisions pending. Any layout change gives every run a new logical digest, and convention receipts bind to that digest. Today no signed receipt would be invalidated.

## Problem

Compact v7 runs store `roi_angles`/`frame_angles` and `roi_qa`/`frame_qa` against one shared channel index per family. Each dense array therefore has a column for every channel on either axis. Channels absent on an axis are placeholders: `NaN` for angles, `0` for QA. Only the index's `frame_available`/`roi_available` flag tells them apart from data.

| Family | Channels in index | Real on frame axis | Frame placeholders |
|---|---|---|---|
| angle | 141 | 100 | 41, including `heading_deg` |
| qa | 7 | 3 (`valid_frame`, `major_axis_marginal`, `reason_codes`) | 4: `valid_left`, `valid_right`, `left_major_axis_marginal`, `right_major_axis_marginal` |

On 2026-10-07 a decoding agent read `frame_qa` by name and reported that per-eye validity was "never valid" in all 84 runs. The data was intact in `roi_qa`, and the supported reader would have hidden those columns. The cause was the layout: a `uint16` `0` placeholder cannot be told apart from "false".

Three production readers had the same bypass. They decoded channel names themselves and never checked availability. They were correct only because the channels they request all happen to be frame-available:

- `analysis/chaser_gaze_tracking.py` (`_packed_columns`)
- `analysis/gaze_convention_validation.py` (`_read_packed_columns`)
- `analysis_workflows/eye_gaze_source_handle.py` (`_decode_names`)

## Decision: no sentinel

A sentinel cannot fix this. QA arrays are `uint16`, so `NaN` is unavailable. Any non-zero sentinel such as `0xFFFF` makes a naive `valid_left != 0` read "always valid", which is worse than "never valid". Only readers that already know about the sentinel would benefit, and those readers could check `frame_available` instead. The fix is for each axis to carry only its own channels.

## Stages

Each stage is a separate change. Per AGENTS.md "Stage adoption, integration, and removal separately".

### Stage 1: one fail-closed column reader (implemented)

Classification: **enforcement correction**, behavior-preserving for valid v7 runs.

- `fisheye.analysis.eye_angle_io.read_compact_axis_columns(run_group, family=, axis=, names=, row_windows=None)` is the single owner of the compact name → column lookup. It refuses:
  - a missing name array;
  - names that are empty, duplicated or count-mismatched;
  - a missing or wrong-length `{axis}_available`;
  - an absent channel;
  - a placeholder channel.
- The three readers above now call it; their private decoders are deleted. Their I/O patterns are kept: an orthogonal column selection for full reads, and full-width row windows for the sampled convention validator.
- Preserved: returned values. On `2026-08-10T17-20-55Z_arena_1` v3, every column used by the three readers is byte-identical to a raw `array[:, index]` read, both for full reads and for row windows. Also preserved: output receipts/digests, since no persisted bytes change.
- Tightened, intentionally: a run without `{axis}_available`, or with duplicate or empty names, is now refused by these readers. Every v7 run written by the current writer carries both flags.

### Stage 2: compact v8 writer with per-axis indexes (proposed)

Classification: **schema change**, versioned. Values are unchanged; layout and digests change.

- `frame_angles`, `frame_qa` and `frame_vectors` get their own index groups that list only frame channels. Proposed names: `frame_angle_channel_index`, `frame_qa_channel_index`, `frame_vector_channel_index`. `roi_*` keeps the existing index groups, which then list only ROI channels. The `*_available` arrays are dropped. A column a reader can resolve is always real.
- Frame column order is the v7 semantic order with placeholders removed. The physical layout doc reserves the first 16-column chunk for frame-available channels, so the interactive core chunk keeps its membership.
- Version bumps:
  - `schema_version` 7 → 8;
  - `compact_dense_arrays` v2;
  - logical equality contract `eye_angle_compact_v7_arrays_v1` → a v8 contract;
  - `EYE_ANGLE_LOGICAL_ARRAY_COUNT` re-derived;
  - storage receipts in `analysis/eye_angle_storage.py`;
  - exact-schema validator in `shared/eye_angle_schema.py`.
- Preservation tests, written before the writer change:
  - for every `(axis, channel)` that is available in v7, the v8 column is byte-identical, using the same source run materialized both ways;
  - every v7 placeholder is absent from v8;
  - malformed, tampered and wrong-version v8 runs are refused.
- `read_compact_axis_columns` and `load_eye_angle_run_tables` read both v7 and v8. v7 stays readable and is not rewritten in place.
- Also fix here: `eye_angle_io._channel_availability` defaults to "all available" when the flag array is missing. That is fail-open, and only v7 still needs a flag. Some existing `test_eye_angle_io.py` fixtures omit the flags and must be updated.
- Give QA channels the same written meaning angle channels already have. Today the angle index stores `representation`, `eye`, `value_kind`, `units`, `source_channel`, `formula` and `compatibility_alias_of` for each channel. The run attrs add `*_definition` strings and `reason_code_map`. The QA index stores only `name`, `value_kind` and `dtype`. Nothing in a run records either rule:
  - `valid_frame = valid_left & valid_right & detection_success`;
  - per-eye angles are `NaN` wherever that eye's ellipse is invalid.

  Add a `formula` column to the per-axis QA indexes, built by `eye_qa_channel_metadata` in `shared/eye_angle_schema.py` like the angle formulas, plus a run-level `per_eye_validity_definition` attr. The attr says per-eye frame validity is `isfinite` of that eye's angle, equivalently `(reason_codes & (eye_bit | 32)) == 0` with `eye_bit` 4 (left) or 8 (right). Bit 32 (`no_detection`) is required: no-detection frames carry no per-eye bit. This was measured with zero mismatches on one real run; a writer-side test must prove it for every run.

### Stage 3: migrate remaining physical readers (proposed)

These read the compact arrays or index groups directly and need to accept v8:

- `analytics_exports/eye_trace_samples.py`
- `analysis_workflows/materializers/eye_angles.py` (completion validator, around line 2350)
- `visualization/visualize_eye_angles.py`
- `analysis/eye_angle_storage.py`
- `diagnostics/benchmark_eye_angle_*`

Add a scoped AST/import check that only `eye_angle_io` and `shared/eye_angle_schema.py` index `frame_angles`/`frame_qa` columns by decoded name.

### Stage 4: default switch and cohort rematerialization (proposed, needs authorization)

- Make v8 the writer default.
- Re-materialize the 84 goodbatbadbat candidates as v8, as fresh selector-ineligible outputs.
- Then run the gaze convention review against the v8 logical digests.
- v7 runs stay readable. Historical runs are not rewritten.

## Decided

- **No per-eye validity channels on the frame axis** (Jeremy, 2026-10-07). Per-eye frame validity already exists twice: `reason_codes` (eye bit 4/8 plus `no_detection` bit 32), and `NaN` in that eye's angles. A third copy could drift from both. It would also need its own detection → frame reduction rule, and no consumer needs it. Stage 2 documents the existing encoding (QA `formula` column and the `per_eye_validity_definition` attr) instead of adding a column. Revisit only if a monocular analysis needs it, and then derive the column from the bits under a test that it matches the `NaN` pattern.

## Open decisions

1. **Separate per-axis index groups (proposed) or one index with per-axis column positions.** Separate groups make a placeholder impossible to express. One index with positions keeps a single name table but leaves room for the same misread.
2. **Rematerialization timing.** Should the 84-run cohort be re-materialized as v8 before the pending convention review starts, so the receipts bind once?
