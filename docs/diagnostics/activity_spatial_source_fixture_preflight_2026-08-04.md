# Activity/Spatial Source Fixture Preflight — 2026-08-04

Status: implementation and disposable full-duration preflight pass. This is not
profile-promotion evidence. A clean-revision fixture and query-export benchmark
remain the next checkpoint.

## Purpose

The selected Sleepyfish compact-v8 swim-bout authority predates the executable
`array_schema_manifest`, so the maintained activity/spatial exporter correctly
refused it. Re-running the historical detector under the current numerical
environment produced the same row count but did not reproduce every persisted
float exactly. The source therefore must not be replaced by a claimed semantic
recomputation.

`fisheye.diagnostics.build_activity_spatial_source_fixture` now creates an
isolated benchmark archive by:

1. copying the exact selected track-motion authority and its transitive internal
   Zarr dependency closure;
2. copying the historical swim-bout run without mutating its source;
3. applying only the closed `lossless_fixed_utf8_width_widening_v1`
   compatibility projection;
4. installing the executable compact-v8 manifest on the isolated copy;
5. proving exact decoded equality for every non-widened array and source-prefix
   plus zero-padding equality for each widened array;
6. validating the exact track, frame-axis, candidate, signal, and bout binding;
7. checking direct/consolidated metadata equivalence; and
8. publishing by one final directory rename under a benchmark-only namespace.

The fixture manifest records Palette commit and dirty state. Dirty builds remain
usable for development but are explicitly evidence-ineligible.

## Compatibility Findings

The untouched historical run contains 132 arrays. Its fixed-text projection is
closed:

- `indexes/candidates/candidate_name`: physical width 64 to 256;
- `indexes/candidates/parameters_json`: physical width 512 to 8192;
- `indexes/signal_variants/parameters_json`: physical width 512 to 8192; and
- `tables/summary_metrics/metric_name`: logical declaration `S36` to `S64`,
  with no physical-array change.

Every source byte is preserved and every added byte is zero padding. No other
physical array may change under this policy.

The selected exponential signal contains 26,565 bouts. Durations are all finite
and nonnegative. Three bout path lengths are IEEE NaN; all finite paths are
nonnegative. Compact swim-bout null semantics already define NaN as unavailable,
so activity source-binding, binning, and export envelopes are now revision 2:

- finite nonnegative duration remains mandatory;
- finite nonnegative path length or NaN is accepted;
- infinity and negative finite path length fail;
- a started NaN path preserves the event count, duration, and occupancy;
- the bin path sum becomes NaN and `bout_metrics_valid=false`; and
- the exporter never drops that bout or sums only its finite peers.

## Disposable Full-Duration Result

The development preflight used:

- recording: Sleepyfish Cam2010095;
- track authority:
  `track_kinematics_sleepyfish_cam2010095_current_coordinates_20260724_v003`;
- historical swim-bout authority:
  `swim_bouts_sleepyfish_cam2010095_exponential_20260724_v003`; and
- output:
  `/tmp/.palette_benchmarks/analytics_query_exports/fixtures/activity_source_dryrun_lossless_v1`.

It passed with:

- 132 declared arrays: 129 exact and three losslessly widened;
- 144 direct/consolidated-equivalent swim-bout nodes;
- 7,066 fixture files;
- 352,485,959 apparent bytes;
- 116.859 seconds before visibility;
- 68.646 seconds for track/dependency projection and validation;
- 12.886 seconds for swim-bout projection and attestation;
- 1.383 seconds for scratch-to-hidden publication; and
- unchanged source track, swim-bout, and projected dependency inventories.

This run preceded the final clean-revision binding and is deliberately retained
only as development preflight. It does not authorize selection, registration,
profile promotion, or canonical-data changes.

## Validation

- 231 combined activity publisher, fixture, benchmark, workflow, and Arrow
  contract tests passed outside the sandbox.
- The final focused fixture/activity/swim-schema gate passed 33 tests outside
  the sandbox.
- Ruff, `py_compile`, Black, and `git diff --check` passed on the changed Python
  surfaces.

## Next Gate

Commit this checkpoint, rebuild the fixture from that clean immutable revision,
then run the maintained full-duration `activity_spatial_time_bins` publication
and read matrix. The result must remain selector-ineligible and nonpromoting.
