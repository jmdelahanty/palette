# Crimson Track Motion Read Contract

Date anchored: 2026-07-19

Status: branch-local future-normal contract for Palette commit `93177ed5`. The
scoped Palette producer/strict reader passed independent review and the 101/101
focused suite, but canonical-v2 is not deployed to current registry archives,
no authoritative cross-repository contract is merged, and Crimson is not
implemented for this future-normal track-motion boundary.

Purpose: define the Palette-side future-normal read contract Crimson should use
for motion traces, swim-bout overlays, and per-bout metrics. This is the
Palette-side future-normal replacement for legacy Crimson
`analysis/movement_runs` loading.

## Source Map

Use these run families for new Crimson readers:

| Need | Preferred Palette source |
| --- | --- |
| Per-track positions with coordinate/identity authority | `analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/` through the strict full-motion loader |
| Heading, speed, path distance, acceleration, and time-series validity | same verified full-motion publication; consume logical manifest surfaces, never paths or aliases alone |
| Swim-bout event windows and detector traces | `analysis/swim_bout_runs/<run>/` through the logical candidate/signal resolver |
| Per-bout physical movement, heading, eye-gaze summaries | `analysis/bout_kinematics_runs/<run>/` |
| Body frame, snout/tail landmarks, B-spline, subject-shape QC | `analysis/subject_shape_runs/<run>/` |
| Protocol step timing and stimulus geometry | `analysis/stimulus_runs/<run>/steps/step_<i>/` |

Legacy `analysis/movement_runs` is available only to explicitly selected
historical inspection or migration tooling. It is not a normal future-recording
read path.

Stimulus step details are specified in
[`crimson_stimulus_step_read_contract.md`](./crimson_stimulus_step_read_contract.md).

## Track Kinematics

`analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/` stores
positions plus gap-aware speed, distance, heading, and validity values. The
strict typed boundary authorizes them only through one closed, freshly
reconstructed full-motion publication seal.

For canonical future runs, Crimson must freshly validate all three gates:

```text
run.attrs["coordinate_binding_status"] == "bound_canonical_v2"
run.attrs["palette_run_completion_status"] == "complete"
run.attrs["stage_selector_eligible"] is True
```

An incomplete, staged, publishing, selector-ineligible, unsupported, or
ambiguously labelled run is not selectable. Convenience `latest*` attrs are
discovery hints only and do not override these gates. After resolving a pointer,
the reader must reopen the exact child and reconstruct its live full-motion
manifest before copying values. It must revalidate the exact child after the
copy as well; lifecycle gates alone do not authorize any array.

Implicit selection additionally requires one stable cross-parent selector
state. For scope `<scope>` and child `<run>`, all four values must agree:

```text
analysis/track_kinematics_runs.attrs["latest"]          == "<scope>/<run>"
analysis/track_kinematics_runs.attrs["latest_complete"] == "<scope>/<run>"
analysis/track_kinematics_runs.attrs["latest_<scope>"]  == "<run>"
analysis/track_kinematics_runs/<scope>.attrs["latest"]  == "<run>"
```

Any mismatch is an in-progress or invalid selector handoff and must return a
retry/fail-closed result. It is not permission to read the older child named by
one surviving pointer. After trimming outer whitespace only, explicit selection
accepts a bare direct-child name or the exact path
`analysis/track_kinematics_runs/<scope>/<run>`. Wrong-scope paths, qualified
shorthand, extra descendants, and other malformed forms are rejected.
Explicit selection does not require the four discovery selectors to agree, but
the exact child must still pass all three lifecycle gates and the full-motion
seal.

Crimson never repairs selector state. On Palette publication failure, selector
restoration is limited to exact mutations in the deferred owner-bound activation
receipt. Palette re-resolves the archive/selector parents and rechecks both the
lease owner and attempted value before every write; it stops on takeover or an
unexpected value. When the exact failed child remains owned, the materialized
publisher retains it failed and selector-ineligible and must persist and verify
an owner-bound tombstone. Failure to prove that state is reported as incomplete
rollback. A takeover or foreign replacement is left untouched. Normal readers
fail closed in every case; Palette never restores a generic pre-copy snapshot
or clobbers a successor.

Canonical runs require:

```text
track_motion_publication_manifest
track_motion_publication_manifest_sha256
track_motion_publication_commit
```

The closed manifest binds the exact run/track/group/array inventories; every
array payload, dtype, shape, attrs, axis-0 domain, identity key, logical
operation, and resolved input; all coordinate, time, source-row, calibration,
and transform records; and the exact run derivation parameters and inputs.
Unknown children, unresolved references, stale attrs, or changed payloads fail
closed. A caller-supplied detached handle is never trusted: the reader rebinds
the authoritative child and then rejects any changed or mixed detached payload.

Required canonical identity and position fields:

```text
track_sample_key                       # int64 [track_id, acquisition_frame]
source_acquisition_frame_index         # authoritative camera-frame mapping
source_frame_interpolation             # exact acquisition-time lineage
source_instance_key                    # nullable observation lineage
source_row_index                       # exact selected immediate-source row
positions_px
```

Current publication-manifest schema v1 requires `frame_indices` as a sealed
exact compatibility alias of `source_acquisition_frame_index`, and the normal
loader validates equality. It is never primary identity: canonical consumers
use `track_sample_key` and `source_acquisition_frame_index`; row offset and
`frame_indices` alone are not identity. A future no-alias schema may omit it
only through an explicit manifest/reader schema-version change.

`positions_mm` and the `*_mm` motion fields are optional. They are present only
when Palette bound an exact compatible typed physical-frame calibration. Their
absence means physical output is unavailable, not that Crimson should infer a
scale. Crimson must never apply a run-level `pixel_to_mm` scalar or a resolution
ratio to create missing physical coordinates.

### Position coordinate contract

Every canonical `positions_px` or `positions_mm` array owns a
`coordinate_descriptor` and digest. Crimson must validate the array-owned
descriptor, its row identity, exact reference-extent/frame records, transform
chain, and derivation records before use. See
[`coordinate_metadata_framework.md`](./coordinate_metadata_framework.md).

For `positions_px`:

- `space_id == "source_camera_image_px"` with
  `source_camera_overlay.status == "direct"` may be drawn directly on the exact
  referenced source-camera frame;
- `source_camera_overlay.status == "requires_transform"` requires applying the
  persisted ordered direction-labelled chain;
- `not_suitable`, an unsupported profile, a stale digest, or missing evidence
  fails closed.

Canonical offline track publication currently requires an exact dtype-preserving
subset/reorder of the selected crop producer's persisted source-camera
`centers_img_xy`. It does not reconstruct positions from normalized detection
centres, root dimensions, ROI dimensions, or simple scale factors.

Historical `coordinate_space = "camera"` or `"texture"` attrs are not
coordinate authority. A legacy reader may resolve them only through an explicit
compatibility mode with exact dimensions and lineage. New recordings and normal
Crimson paths do not use that adapter.

Renderer viewport/display coordinates remain ephemeral Crimson state. They are
not written back as Palette coordinate metadata.

The existing grouped motion layout is:

```text
movement/speed/<raw|filtered|smoothed|averaged>/
  px
  acceleration_px
  smoothed_acceleration_px
  mm                              # optional physical peer
  acceleration_mm                 # optional physical peer
  smoothed_acceleration_mm        # optional physical peer

movement/speed/<raw|filtered|smoothed>/
  frame_path_distance_px
  frame_path_distance_mm          # optional physical peer
```

All four levels retain acceleration and smoothed-acceleration peers; only
`averaged` omits frame-path-distance peers. Every listed speed and derivative
record uses `axis0_domain == "track_transition_destination_sample"`: row `i`
describes the transition ending at sample `i`. Averaged values remain temporal
averages of those destination-anchored transition values; they are not ordinary
`track_sample` values.

This list describes current publication-manifest schema v1's physical layout
only; it is not coordinate or derivation authority. The normal v1 reader
resolves and copies only exact sealed grouped logical records. It never searches
flat speed/path arrays or `speed_derivatives` as alternatives. The v1 writer and
seal still physically require and validate their closed compatibility aliases,
and some grouped derivative records are themselves digest-bound exact aliases;
those relationships are seal obligations, not fallback permission. Missing or
tampered v1 aliases invalidate the seal while Crimson still never reads them as
alternate logical targets. Only
`load_legacy_track_kinematics_track_for_inspection` performs physical fallback.
The deferred
[`future_track_motion_storage_layout.md`](./future_track_motion_storage_layout.md)
must version the compact no-alias manifest and reader rather than add runtime
negotiation to v1.

`source_acquisition_frame_index` is the authoritative sparse row-to-camera-frame
lineage. Crimson must not assume track row index equals video frame index. Build
a frame-to-row lookup for interactive seeking and cross-check it against column
1 of `track_sample_key`. Under manifest schema v1, `frame_indices` must exist as
the sealed compatibility alias and be numerically identical, but Crimson must
not use it instead of the two authoritative fields.

Missing frames are gaps. Displaying them as zero speed is a UI choice, not the
stored analysis semantics.

## Swim-Bout Runs

`analysis/swim_bout_runs/<run>/` is the canonical bout segmentation candidate
surface. It answers: what events did this detector or speed level find?

Crimson should treat swim-bout data as a logical run/candidate/signal surface,
not as a fixed physical path. Palette currently has two physical layouts:

- hierarchical v1: `analysis/swim_bout_runs/<run>/<speed_level>/...`
- compact v2: `analysis/swim_bout_runs/<run>/indexes/*`,
  `tables/*`, and `signals/*`, with `attrs["layout"] ==
  "compact_tabular_v2"`

The detailed compact-v2 handoff is in
[`crimson_swim_bout_compact_v2_read_contract.md`](./crimson_swim_bout_compact_v2_read_contract.md).

Important fields:

```text
bouts
bout_points
inter_bout_intervals
global_metrics
detection_signal_mm_s        # present for transformed detector responses
speed_exponential_mm         # compatibility/plotting mirror when present
```

In compact v2, these logical fields are materialized differently:

```text
indexes/candidates
indexes/signal_variants
tables/bouts
tables/peak_events
tables/inter_bout_intervals
signals/detector_signal_mm_s
signals/detector_signal_signal_ids
signals/frame_indices              # historical or declared portable fallback
```

Crimson should create one selectable UI candidate per selected candidate row
and signal row, using `candidate_id` and `signal_id` as stable identity inside
the run. Do not require a physical `<speed_level>` subgroup when
`layout == "compact_tabular_v2"`.

For detector traces, new run-schema-8 outputs resolve the frame axis from the
versioned run-level `frame_axis_contract`. Existing schema-8 outputs may point
to the exact source track-kinematics compatibility `frame_indices` array
relative to the same Zarr root; that source array must itself declare
`source_acquisition_frame_index` as authority and match it exactly. Schema-7
runs continue to use embedded `signals/frame_indices`, and schema-8 runs may
declare that same path as an embedded portability fallback.
Use the resolution and fail-closed checks in the detailed compact-v2 handoff;
do not substitute a `latest` track pointer or reconstruct an `arange` axis.

The `bouts` table includes frame and timing boundaries such as:

```text
start_frame
end_frame
core_start_frame
core_end_frame
start_time_s
end_time_s
duration_s
observed_duration_s
path_length_mm
path_length_px
net_displacement_mm
net_displacement_px
peak_detection_signal_mm_s
peak_physical_speed_mm_s
```

Frame boundaries are authoritative for overlay rectangles and slicing. Optional
interpolated threshold times are annotations, not replacements for frame
boundaries.

### Detector vs Physical Metrics

`speed_exponential` is a detector response, not an independent physical speed
measurement. It can be useful for bout segmentation and visualization, but
biological speed, path length, and active-duration metrics should be read from
declared physical movement sources such as `speed_filtered` or from linked
`analysis/bout_kinematics_runs`.

Crimson should label detector traces as detector responses when plotted.

## Matching Track Kinematics To Swim Bouts

Given a selected track-kinematics run, track ID, and speed level, Crimson should
discover compatible swim-bout candidates by lineage attrs and/or compact signal
columns:

```text
source_track_kinematics_run
track_id
detection_signal_source_level
detection_signal_source_path
movement_metric_source_level
source_level
path_distance_source_level
signal_id
candidate_id
```

Direct matches:

- selected `filtered` speed maps to subgroup `speed_filtered`
- selected `smoothed` speed maps to subgroup `speed_smoothed`

Transformed matches:

- `speed_exponential` is compatible only when its attrs point back to the
  selected source speed, for example
  `detection_signal_source_level = "filtered"`.

Do not auto-select a bout subgroup whose detector source points at a different
speed trace than the selected track-kinematics speed.

## Bout Kinematics

`analysis/bout_kinematics_runs/<run>/` is downstream of one exact
track-kinematics run and one exact swim-bout candidate. It owns physical
per-bout measurement policy.

Use it for:

- physical active duration
- physical active path length
- physical active mean and peak speed
- pre/post position and heading windows
- within-bout heading changes
- optional eye-gaze summaries

Primary group:

```text
movement/per_bout_metrics/
```

Key fields include:

```text
source_start_frame
source_end_frame
source_core_start_frame
source_core_end_frame
physical_active_start_frame
physical_active_end_frame
physical_active_duration_s
physical_active_duration_s_interpolated
physical_active_path_length_mm
physical_active_path_length_px
physical_active_mean_speed_mm_s
physical_active_peak_speed_mm_s
physical_active_valid
failure_reason_bytes
```

Crimson should not mutate swim-bout runs when displaying these metrics. Bout
kinematics is a linked measurement layer, not a replacement segmentation layer.

## Subject Shape Boundary

`analysis/subject_shape_runs/<run>` is the geometry/QC surface for body axes,
B-splines, snout/tail landmarks, and body-frame overlays. It is not the default
source for speed or path distance and must not be used to bypass a missing
track-motion payload seal.

Subject-shape fallback motion labels are acceptable only as a preview/debug path
for archives missing `analysis/track_kinematics_runs`. If used, label them as
derived preview values.

## Recommended Crimson UI Slice

1. Discover `analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/`
   and validate its typed position binding.
2. Let the user select a track. Expose speed-level selection only after the
   exact derived-motion payload/derivation seal is supported and validates.
3. Build a sparse frame-to-track-row lookup from
   `source_acquisition_frame_index`, cross-checked against `track_sample_key`.
4. Draw the validated position in its declared overlay frame. After the exact
   full-motion seal validates, draw per-frame motion labels:
   - heading from `heading_degrees` or `smoothed_heading_degrees`
   - speed from the selected sealed logical record; use its validated mm/s peer
     when present, otherwise report the validated px/s record with honest units
   - never substitute flat or `speed_derivatives` paths for a missing grouped
     logical record
5. Discover matching `analysis/swim_bout_runs` candidates and overlay bout
   windows from logical `bouts/start_frame` and `bouts/end_frame`, regardless
   of whether the run is hierarchical v1 or compact v2.
6. Load linked `analysis/bout_kinematics_runs` for per-bout measurement tables
   and histograms when present.
7. Keep subject-shape overlays independent from motion traces.

## Focused conformance cases

- Independently mismatch each of the four implicit selectors and require
  retry/fail closed; prove that an explicitly pinned exact child is unaffected
  by unrelated selector drift.
- Reject wrong scope, extra suffix, qualified shorthand, repeated/traversal
  segments, and every normalization trick beyond surrounding-whitespace trim.
- Leave flat aliases intact while mutating or removing a grouped target and
  require the normal reader to fail rather than fall back; the explicitly named
  historical loader remains unverified inspection only.
- Require averaged acceleration and smoothed-acceleration peers, forbid averaged
  path-distance peers, and validate destination-transition domain on every
  averaged logical record.
- In Palette producer-only fixtures, cover an intervening winner before lease,
  takeover during rollback, and exact-receipt rollback failure; none may restore
  a stale pre-copy snapshot or clobber a successor.

## Canary

Historical/inspection feeding canary unless independently proven canonical-v2;
it is not release evidence for this future-normal reader:

```text
/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr
```

Useful historical-inspection sources:

```text
analysis/track_kinematics_runs/offline/tk_hyst4_low2_s005/tracks/id_0
analysis/swim_bout_runs/<candidate>/<speed_level>
analysis/bout_kinematics_runs/<candidate>
analysis/subject_shape_runs/<candidate>
```

The archive has no useful `analysis/movement_runs` path for historical Crimson
motion inspection.
