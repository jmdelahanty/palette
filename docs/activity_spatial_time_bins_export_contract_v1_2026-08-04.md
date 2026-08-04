# Activity And Spatial Time-Bin Export Contract V1

Date: 2026-08-04

Status: exact logical/Arrow schema, per-track bout-source binder, and pure bin
aggregator implemented; bounded publisher and promotion evidence remain
unimplemented. This decision activates no workflow default, selector, registry
authority, Zarr physical profile, or canonical-data change.

## Decision

`activity_spatial_time_bins` is an immutable portable Parquet query product.
The recording-local scientific authorities remain one explicit completed track
motion run and an explicit completed swim-bout run for every exported track.

The workflow currently binds track motion and bouts, but it does not bind an
arena or experimental-area geometry authority. Consequently v1 reports
physical-position distributions and track coverage; it does not invent wall,
center, arena-normalized occupancy, quadrant, or accessible-area metrics.
Those remain owned by the recording-local detection/session occupancy families
until a future export explicitly binds their geometry manifests.

## Grain And Identity

There is one row for each explicit source-run pair, track, and global
acquisition-frame-aligned time bin. The exact key is:

```text
(recording_id,
 source_track_kinematics_scope,
 source_track_kinematics_run,
 source_swim_bout_run,
 track_id,
 time_bin_index)
```

Rows sort by track ID and then time-bin index. The maintained compact swim-bout
schema currently owns one track per run: its candidate/signal indexes do not
carry track identity, while the run and bout rows do. A multi-track export must
therefore receive an exact `track_id -> swim_bout_run` mapping. Each run must
provide one selected default candidate and signal for its declared track.
Within-frame ordinals, implicit track zero, and one run silently reused across
tracks are forbidden.

## Global Binning

For source frame rate `fps` and requested positive bin width `w`:

```text
bin_size_frames = max(1, floor(fps * w + 0.5))
effective_bin_size_s = bin_size_frames / fps
time_bin_index = source_acquisition_frame_index // bin_size_frames
```

Bin boundaries never restart at a clip, track, source read window, or worker.
For each track, rows cover every global bin intersecting its inclusive first to
last acquisition-frame span. Empty internal bins are emitted so tracking gaps
remain visible. `expected_track_frame_count` is the intersection between that
global bin and the track span; coverage fractions use this denominator.

## Metrics

The exact 70-field Arrow schema carries:

- closed track-motion and swim-bout source identities and digests;
- physical-coordinate authority and exact binning policy;
- observed, sample-valid, position-valid, and transition-valid counts and
  fractions;
- mean, population standard deviation, covariance, extrema, and endpoint net
  displacement for valid physical-mm positions;
- mean, median, and p95 filtered speed plus valid transition path distance;
- bouts assigned by start frame, their duration/path sums, and the union of
  inclusive bout-frame intervals overlapping each bin; and
- independent position, speed, bout, and overall-bin validity fields.

Position metrics use rows where both `sample_valid` and `position_finite` are
true. Speed/path metrics use finite filtered values with `transition_valid`.
Invalid floating metrics are IEEE NaN, never Arrow nulls or numeric sentinels.

`bout_count_started`, duration, and path sums assign a whole bout to the bin
containing its `start_frame`. `bout_occupied_frame_count` instead clips every
inclusive `[start_frame, end_frame]` interval to the bin and counts the union,
so a bout spanning two bins contributes occupancy to both without double
counting overlapping frames. These two allocation rules are deliberately
separate and named.

## Source And Publication Boundary

The publisher must fail closed unless it can prove and recheck:

- the exact completed, selector-eligible track-motion manifest/commit and
  physical-mm position authority;
- each exact completed, selector-eligible swim-bout array manifest;
- exact track-run, track-ID, frame-axis, candidate, signal, and speed-level
  lineage between the two sources;
- one explicitly mapped run and one default bout candidate/signal per exported
  track; and
- unchanged source manifests, completion state, and selection snapshots before
  the manifest-exclusive export becomes visible.

The future publisher must stream bounded source windows, construct parts on an
explicit non-overlapping node-local scratch root, hash decoded projected
columns, copy into a hidden generation, and publish only through the existing
manifest-exclusive compare-and-swap boundary. The compact in-memory exporter
already rejects this dedicated table name.

## Deferred Geometry Extension

A future arena-aware profile may add a separate table for normalized occupancy,
wall/center use, or spatial-cell distributions only after binding:

- one exact experimental-area geometry and coordinate transform;
- accessible-area cell weights and grid policy;
- geometry identity compatible with every track position; and
- a versioned boundary-distance method.

That extension must use a new table/schema version. It must not reinterpret v1
position moments as arena occupancy.

## Implementation Surface

- `src/fisheye/analytics_exports/activity_spatial_time_bins.py`
- `src/fisheye/analytics_exports/contracts.py`
- `src/fisheye/analytics_exports/arrow_contracts.py`
- `src/fisheye/analytics_exports/capabilities.py`
- `src/fisheye/analysis_workflows/profiles/core_behavior_v1.yaml`

## Remaining Gates

- Add bounded source-window reads, manifest-exclusive publication, full
  decoded validation, tampering, and
  interrupted-replacement recovery tests.
- Wire the workflow/LSF node-local execution boundary.
- Benchmark short and full-duration writer/read/copy/validation/publication
  behavior before considering default activation.

## Validation Evidence

The source binder requires an exact run map, maintained schema/manifest,
completion and eligibility, track-motion manifest equality, canonical frame
axis, one default candidate/signal, and selected event-content digest. The pure
aggregator covers global bin rounding, empty internal gaps, clipped edge
denominators, start-assigned bout metrics, and interval-union occupancy. The
focused Arrow/source-binder/aggregation/workflow matrix passed 214 tests; Ruff,
Python compilation, and `git diff --check` passed.
