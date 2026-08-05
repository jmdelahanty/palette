# Activity/Spatial Representative-Short Contract

Date: 2026-08-04

Status: implemented and unit-validated; the real clean-revision matrix has not
yet run. No selector, registry, storage profile, or source archive is changed by
this checkpoint.

## Goal

The representative-short activity/spatial benchmark must measure a real
200,000-frame scientific interval, not publish the full recording and label a
short reader query as a short workflow. The bounded export remains an immutable
projection of the accepted full-duration track and bout authorities.

## Frozen interval

- `representative_short` requires exactly one explicit half-open acquisition
  frame interval `[source_frame_start, source_frame_stop_exclusive)`.
- Its length must be exactly 200,000 frames.
- `full_duration` rejects either interval field.
- Both fields are non-negative integers, must be supplied together, and must
  define a non-empty interval.
- The selected track span is the requested interval intersected with the
  source track's inclusive frame span.

The benchmark request transports this interval unchanged through the direct
activity exporter and through the atomic publisher callback.

## Version boundary

The existing unbounded contracts remain readable and unchanged:

- activity/spatial binning schema v2;
- activity/spatial export schema v3.

The bounded projection uses:

- activity/spatial binning schema v3;
- activity/spatial export schema v4.

The v3 binning digest binds the requested interval and the exact edge,
denominator, occupancy, started-bout, validity, and selected-row policies. The
v4 export envelope mirrors both interval endpoints and requires the v3 binning
contract. A bounded envelope cannot be relabeled as unbounded, or vice versa,
without invalidating its digests.

## Bin and bout semantics

Global bin identity does not change. Each emitted row keeps the full global
bin start, end, time, and duration values. Scientific aggregation is clipped to
the selected span:

- expected-frame denominators use `bin ∩ selection ∩ track`;
- track row bounds and validity use the same clipped span;
- occupancy is the union of bout intervals clipped to that span;
- a bout contributes its whole duration/path metric only when its start frame
  lies inside that span;
- a bout that starts before the selection can contribute occupancy but cannot
  be counted as a newly started bout;
- a bout starting at the exclusive stop is excluded;
- the final edge bin is allowed to be partial.

A requested interval disjoint from the track authority is a valid zero-row
projection. Its Parquet file still has the exact schema and validated footer,
but naturally has no data pages whose encodings could be inspected.

## Source identity and cost boundary

The bounded writer continues to validate and hash the complete selected source
authorities. It does not replace full-source identity with a hash of only the
200,000-frame slice. That makes the first short-workflow publication timing
conservative, but preserves a strong immutable source binding.

The bounded multi-bin source-read policy remains in force, so interval
selection does not reintroduce the former one-read-per-bin amplification.

## Validation completed

- [x] Exact interval construction and digest coverage.
- [x] Missing-pair, boolean, negative, and empty-interval rejection.
- [x] Non-bin-aligned edge-bin behavior.
- [x] Clipped denominators and union occupancy.
- [x] Bout-start behavior before, inside, and at the exclusive boundary.
- [x] Valid zero-row disjoint projections.
- [x] Binning-v3/export-v4 pairing and tamper rejection.
- [x] Benchmark request and CLI transport.
- [x] Exact 200,000-frame representative-short enforcement.
- [x] Full-duration bounded-request rejection.
- [x] Focused exporter/controller tests: 51 passed.

## Remaining evidence

- [ ] Commit this implementation on a clean immutable revision.
- [ ] Run five fresh representative-short writer/read processes from that
      revision using node-local scratch and immutable benchmark publication.
- [ ] Record writer phase timing, apparent and allocated bytes, object count,
      CPU, peak RSS, random/window/full-scan latency, and logical digests.
- [ ] Independently compare the scientific columns with a recomputed
      `[start, stop)` projection. Do not infer equality by slicing the prior
      full-duration Parquet file because the non-aligned edge bin has different
      clipped denominators and aggregates.
- [ ] Keep every result selector-ineligible and promotion-ineligible until the
      wider analytics promotion gates pass.
