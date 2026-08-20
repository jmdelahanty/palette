# Chaser stimulus-to-camera temporal projection audit

<!-- contract-meta
status: investigation
last_updated: 2026-08-20
implementation: partial
producer_clarification: received_unsupported_for_current_recordings
-->

Purpose: record how existing Palette and Crimson paths handle multiple native
stimulus states that map to one camera acquisition frame, identify the
statistical consequences, and define the evidence required before Palette
publishes a common camera-frame chaser-relative product.

This note is an audit, not a promotion decision. A read-only acquisition-side
audit has now established that current recordings do not persist enough
evidence for an exposure-aligned stimulus projection. Its findings are recorded
below; the exact Citrus producer commit remains unavailable for the inspected
recording because its H5 did not persist `software_version`.

## The two row domains are both valid and are not interchangeable

The canonical stimulus rowset and the camera-acquisition rowset answer
different questions:

1. A **native stimulus-state sample** records one Citrus CPU-side stimulus-state
   update. It is not proof that the corresponding raster was selected or
   physically displayed. Its identity is `stimulus_state_key`, including
   `stimulus_frame_num` and `chaser_index`.
2. A **camera-acquisition frame** records one fish image at one acquisition
   time. Its identity is the sealed acquisition-frame/track-sample authority.

At a stimulus rate near 120 Hz and a camera rate near 100 fps, more than one
stimulus state can legitimately carry the same
`source_acquisition_frame_index`. Those rows are not duplicate stimulus
states. However, joining every one of them to the fish position from that
camera frame repeats the same fish observation and behavioral timepoint.

For two chasers, one camera frame with two mapped stimulus samples produces
four frame-by-stimulus-by-chaser relation rows while still containing only one
fish image. Row count is therefore not camera-frame count, exposure time, or
an independent biological sample count.

## Producer audit verdict for current recordings

The proposed policy
`latest_presented_at_or_before_camera_acquisition_v1` is unsupported for
current recordings.

The acquisition audit reports that `timestamp_ns_session` is sampled from
Citrus `std::chrono::steady_clock` after CPU-side chaser update/render enqueue
and protocol update. The display thread later selects a ready arena buffer,
composites and uploads it, calls `glfwSwapBuffers`, and eventually reaches
projector scanout. No stimulus-state-bound swap, presentation, or scanout
timestamp is persisted. Asynchronous CUDA readiness also means the logged CPU
state can be newer than the raster buffer that was selected for display.

Orange's authoritative camera timestamp is a hardware/PTP timestamp copied
from `CEmergentFrame.timestamp`. Its exact exposure reference point—start,
midpoint, or end—is not declared in the inspected producer, SDK header, or
available vendor description. Citrus's session-relative steady clock and the
camera PTP clock have no persisted checksummed clock transform and therefore
cannot be compared directly.

The sealed mapping

```text
source_acquisition_frame_index = recording_frame_id - 1
```

means only that the named Orange acquisition was the current input provenance
when Citrus produced the state row. Orange assigns and sends that recording
identity after camera frame receipt; Citrus asynchronously retains the newest
received identity. A state carrying acquisition index `i` was therefore
computed after acquisition `i` had already returned. It is not evidence that
the state was displayed before, during, or immediately after exposure `i`.

When several state rows share one source acquisition index, all were produced
while the same Orange input identity remained current. They retain valid CPU
production order, but their presentation order and relationship to later
camera exposures are unknown. Some logged CPU states might never have become
the selected display raster.

## Observed existing behavior

### Palette provider-position candidate: preserve every native sample

`fisheye.analysis.provider_chaser_distance_candidates` preserves each unique
stimulus frame and joins its fish-side values through exact
`source_acquisition_frame_index`. Fish position, validity, provider row, and
motion lineage are repeated when multiple stimulus samples map to the same
camera frame.

The candidate records:

```text
preserve_unique_stimulus_frame_num_then_join_exact_source_acquisition_frame_index_v1
```

Its traces and distributions are stimulus-sample products. Their denominator
must not be labeled as unique camera frames, fish observations, exposure
frames, or independent observations. Their distance values relate a logged
Citrus state to the fish observation that supplied its input-acquisition
provenance; they are not proven simultaneous fish-to-displayed-stimulus
distances.

The read-only Marimo canary follows the same rule. At a bout onset it retains
all stimulus samples mapped to that acquisition frame and reports their count,
indices, distances, and circular bearing summary. It does not choose one
sample, and it emits one bout/chaser summary row rather than one bout row per
mapped stimulus sample.

### Historical Palette camera-frame chaser distance: implicit final-row wins

`fisheye.analysis.chaser_distance_runs` allocates one slot per camera frame.
Both frame metadata and chaser positions are assigned in source-row order.
Later rows overwrite earlier rows for the same camera-frame/chaser slot.

When input rows happen to be ordered by increasing stimulus frame, this acts
like `highest_stimulus_frame_num_wins`. That is an implementation consequence,
not a timestamp-bound physical policy. The historical product does not record
enough evidence to reinterpret final-row selection as confirmed display state.

### Strict Palette coordinate publication: reject ambiguous collapse

`fisheye.analysis.chaser_distance_coordinate_publication` rejects a second
valid chaser row for one acquisition-frame/chaser identity. It also rejects
different scalar stimulus-frame or timestamp values assigned to one camera
frame. `fisheye.analysis.chaser_metrics_loader` similarly rejects duplicate
selected canonical rows for one camera frame.

This is fail-closed behavior. It prevents a reader from silently inheriting
the historical overwrite policy, but it means a native many-to-one stimulus
rowset needs an explicit projection before it can enter a camera-frame product.

### Crimson Zarr playback: consume an already collapsed mapping

Crimson prefers the dense
`frame_alignment/camera_to_stimulus_frame_corrected` array. It directly indexes
that array by camera frame and receives one stimulus frame number. If the
direct array is unavailable, it follows one camera-to-metadata-row mapping and
then reads one stimulus frame number from that row.

Crimson therefore does not preserve all native stimulus alternatives in its
raw-video-to-stimulus-video playback decision. It consumes a mapping that has
already selected one stimulus frame.

For raw chaser-state overlays grouped directly by camera frame, Crimson keeps
the greatest `stimulus_frame_num` for each chaser and uses the greatest
`timestamp_ns_session` as a tie-breaker. Bounding-box overlays similarly prefer
the greatest stimulus frame. These are useful deterministic display policies,
but the viewer does not establish that the retained state was physically
present during the camera exposure.

The legacy Crimson H5 compatibility path returns the first matching metadata
row when several stimulus frames share one camera frame. The implementation
itself marks this as order-dependent. It must not define the new scientific
policy.

## Statistical consequences of treating every joined row as behavior

Preserving every native stimulus state is necessary provenance. Treating every
joined state as another fish observation would cause several problems:

1. **Repeated behavioral observations.** The same fish position, body frame,
   speed, bout state, and acquisition timestamp appear more than once.
2. **Unequal camera-frame weighting.** Frames with two mapped stimulus states
   contribute twice as much as frames with one, even though both represent one
   camera exposure.
3. **Misleading sample counts.** A raw relation-row count can overstate unique
   camera timepoints and can be multiplied again by the chaser axis.
4. **Pseudoreplication.** Treating joined rows as independent observations
   understates uncertainty. Camera frames are already temporally correlated;
   repeating a frame makes that problem worse.
5. **Invalid transition calculations.** Consecutive stimulus rows can share
   one acquisition timestamp. Computing fish speed, acceleration, heading
   change, entry, or bout transition over that row order can create zero-time
   transitions or count one behavioral transition more than once.
6. **Biased occupancy and dwell denominators.** Sample-count occupancy weights
   states by stimulus-update cadence rather than unique camera exposure or
   elapsed time.
7. **Event duplication.** One bout, visit, trial boundary, or escape can attach
   to several native stimulus rows unless attachment is reduced to one event
   identity with all supporting sample IDs retained as evidence.
8. **False simultaneity.** The source acquisition occurred before Citrus
   produced the state. A joined row must not be interpreted as the fish and a
   physically displayed chaser observed at one timepoint.

These issues can affect descriptive summaries even when inferential statistics
correctly use animal or recording as the biological unit.

## Required separation of products

Palette should keep two explicit products rather than discard provenance or
pretend the axes are equivalent.

### Native stimulus evidence

The native product retains every stimulus state and its exact mapping:

- `stimulus_state_key` and `stimulus_frame_num`;
- stimulus timestamp and source row;
- `source_acquisition_frame_index`;
- chaser identity, role, position, and validity; and
- any fish-side acquisition row used for an exploratory join.

This product may support controller-state/process questions. If it computes a
distribution, its denominator is native CPU state samples or declared CPU-state
time, never camera frames or physical stimulus exposure. Repeated fish-side
fields must be declared as input-provenance joins rather than new observations.
Until a presentation authority exists, the native timestamp can order CPU
state production but cannot establish physical stimulus exposure duration.

### Camera-frame behavioral projection

Behavioral position, motion, occupancy, bouts, visits, and event attachment
would use one row per exact acquisition-frame/track-sample identity only if a
versioned projection can choose or summarize the physically applicable native
stimulus state using producer-supported timing semantics. That projection
cannot be produced for current recordings.

The projection receipt must retain all contributing native stimulus sample IDs
and record:

- policy ID and version;
- camera timestamp authority and exposure reference point;
- stimulus timestamp authority and presentation meaning;
- comparison and boundary rule;
- selected sample ID, if the policy selects one state;
- all contributing sample IDs and count;
- tie, missing, gap, and invalid-state decisions; and
- maximum timing separation or another policy-specific uncertainty measure.

No downstream consumer should recover a policy by sorting rows and taking
`first`, `last`, `min`, or `max` itself.

For current recordings, a product that requires **physical presentation
alignment** must emit unsupported/null with reason
`presentation_time_unavailable`. A separately declared exploratory proxy may
still publish one row per input acquisition, but it must not describe that row
as the stimulus displayed during the exposure or carry a logged state forward
as if it were known to remain on screen.

An interval-integrated stimulus-exposure product may be useful later, but it
would be a third explicitly named product. It must not be implemented by
duplicating a camera fish observation for every stimulus update.

## Count and denominator contract

Every relative or reduced product should expose the relevant distinct counts:

- native stimulus sample count;
- unique acquisition-frame count;
- valid fish-observation count;
- frame-by-chaser relation-row count;
- unique biological subject and recording count at cohort level;
- number and fraction of camera frames with zero, one, or multiple mapped
  stimulus samples;
- maximum native-sample multiplicity per camera frame; and
- the exact denominator and weighting policy used by each summary.

Use `support_row_count` only for literal storage support. Scientific labels
must name the axis, such as `native_stimulus_sample_count`,
`unique_camera_frame_count`, `valid_camera_time_s`, `bout_count`, or
`recording_count`. `valid_exposure_time_s` is permitted only when a physical
presentation/exposure authority exists. None of those counts should be labeled
simply `n` without a declared unit.

For inferential cohort statistics, the biological and experimental unit
remains explicit—normally animal/recording, or a declared repeated-measures
trial or bout hierarchy. Native stimulus rows and camera frames are dense
within-recording observations, not independent animals.

## Defensible current-recording policy

The strongest supportable native policy is:

```text
all_native_states_by_input_acquisition_provenance_v1
```

For acquisition index `i`, it returns every native state whose sealed
`source_acquisition_frame_index == i`, preserving source-row order and
`timestamp_ns_session`. The association means only "computed while acquisition
`i` was the current Orange input provenance." It does not select first, last,
nearest, or interpolated state and does not carry a state forward as a claimed
display exposure.

## Accepted use of explicit proxy analyses

Current recordings may continue to support chaser-relative analyses as
explicit exploratory/controller proxies. This is a separate scientific mode,
not a fallback from unavailable physical presentation alignment.

Each analysis profile or request must declare one of:

```text
temporal_alignment_requirement = physical_presentation_required
temporal_alignment_requirement = input_provenance_proxy_allowed
```

The physical mode remains unavailable for current recordings. The proxy mode
may complete operationally while carrying all of these literal declarations:

```text
temporal_alignment_class = controller_input_provenance_proxy
physical_presentation_verified = false
presentation_timestamp_available = false
camera_presentation_clock_transform_available = false
camera_exposure_reference = unknown
scientific_use_class = exploratory_proxy
```

There is no silent fallback between modes. A module requested with
`physical_presentation_required` remains blocked; the same algorithm may run
under `input_provenance_proxy_allowed` only in a separately identified output
whose manifest, plots, tables, registry status, and cohort export preserve the
proxy class.

The first one-row proxy is now implemented as a pure, read-only computation:

```text
latest_logged_cpu_state_per_input_acquisition_proxy_v1
```

For each input acquisition and complete chaser sample, it selects the
greatest `timestamp_ns_session`, then the greatest `stimulus_frame_num`, with
source-row identity as a deterministic final tie-breaker. It preserves
all candidate state IDs and multiplicity in typed arrays, never carries a state
across an acquisition with no native row, and uses unique input-acquisition
frames—not native state rows—as the behavioral denominator.

That proxy has a useful control-loop interpretation: it relates the fish input
from acquisition `i` to the latest logged controller state produced while `i`
remained current. It still cannot establish what the fish saw during exposure
`i`, and escape, bearing, gaze, distance, pursuit, or near-field results must be
labeled accordingly.

The selector implementation lives in
`fisheye.analysis_workflows.chaser_input_provenance_proxy`. It requires an
already verified native stimulus source handle, rechecks that handle before
selection, retains every candidate native sample and source-row identity, and
emits one selected complete all-chaser sample or an explicit unselected row for
each represented input acquisition. Its compact projection record binds the
exact source manifest and verified-handle snapshot digests.

The result can be published as one exact named, immutable,
selector-ineligible run under
`analysis/chaser_input_provenance_proxy_runs/<run>`. The typed schema preserves
candidate multiplicity and selected same-sample lineage; the strict reader
verifies direct/consolidated metadata agreement, every declared array digest,
source identity, projection semantics, and the absence of a parent selector.
The CLI supports an explicit no-write plan and apply operation. The common
relative-frame publication context accepts only an exact proxy publication
binding and preserves both the readable projection record and its digest.

The keyed adapter is implemented in
`fisheye.analysis_workflows.chaser_proxy_relative_frame_adapter`. It reopens
the exact native candidate bound by the proxy, verifies the native and proxy
authority digests, and transforms selected chaser positions only through the
published typed chain
`arena_relative_canvas_px -> selected_canvas_px -> source_camera_image_px`.
It does not relabel arena coordinates, apply a presentation reflection, guess
a Y flip, or expose Citrus timestamps as camera timestamps. Missing complete
chaser samples remain invalid and are not carried across frames. The output is
an immutable, selector-ineligible common relative-frame candidate whose
context preserves the readable arena geometry, transform chain, proxy
publication, timing, subject, and profile bindings.

An explicit three-job candidate DAG now orders exact native-to-proxy
publication, coordinate-safe relative-frame publication, and a digest-bound
applicability/readiness receipt. That receipt records
`production_authority=false`, `production_selector_activation=false`,
`registry_update=false`, and `physical_presentation_verified=false`. This is
candidate execution evidence, not the standard production chaser DAG. Profile
module execution, scientific successors, recording-local offers, SQLite
projection, production selector activation, and cohort publication remain
open and require their own promotion decision and green required CI.

Future exact presentation-aligned products publish immutable successors under
a different temporal-alignment class. They do not rewrite or silently promote
the proxy outputs. A lag-sensitivity analysis over explicitly declared frame
shifts may be useful for current recordings, but each lag is another named
proxy policy and not an estimate of presentation truth.

Until future recordings provide presentation evidence:

- preserve all native stimulus states;
- keep native-sample products labeled as such;
- do not promote the historical overwrite, Crimson display, or legacy H5
  first-match behavior into a scientific projection policy;
- do not run camera-frame transitions over repeated native-sample rows; and
- keep physical-presentation selectors unavailable; and
- allow only explicitly requested, provenance-complete proxy publications and
  selectors that cannot satisfy physical-presentation requirements.

## Metadata required for future exposure-aligned projection

Future recordings require at least:

- a stable presentation sequence ID binding exact chaser-state rows to the
  exact raster buffer;
- a per-display-frame presentation timestamp from a declared authority,
  preferably page-flip/vblank evidence or a camera-visible photodiode;
- a checksummed transform between presentation time and camera PTP time,
  including scale, offset, validity interval, drift, and uncertainty;
- a declared camera timestamp reference point plus exposure duration or
  explicit exposure start/end;
- live frame-bound recording identities rather than an asynchronously retained
  latest acquisition identity;
- explicit invalid, dropped, and never-presented state markers;
- presentation sequence as the equal-timestamp tie-breaker; and
- a persisted Citrus producer commit or executable digest.

Only then could a future policy select the maximum valid presentation sequence
whose presentation time is at or before a declared camera exposure reference.

## Source locations inspected

Palette worktree `agent/palette/provider-comparison-canary-20260818`, commit
`80e2c2c728b5`:

- `src/fisheye/analysis/provider_chaser_distance_candidates.py`;
- `apps/marimo/components/provider_chaser_candidate.py`;
- `src/fisheye/analysis/chaser_distance_runs.py`;
- `src/fisheye/analysis/chaser_distance_coordinate_publication.py`; and
- `src/fisheye/analysis/chaser_metrics_loader.py`.

Crimson checkout observed at commit `acce5922eb0d`:

- `src/zarr_loader_stimulus.cpp`;
- `src/zarr_loader.h`; and
- `src/h5_loader.cpp`.

The Crimson checkout contained unrelated local changes during inspection. The
two implementation files containing the Zarr and legacy H5 selection behavior
were not modified in that working tree; this note does not promote the Crimson
checkout to a release authority.

The acquisition-side findings above were supplied in the read-only report
`/home/delahantyj@hhmi.org/response.txt`. That report inspected Citrus branch
`isolation` at `31779f418fd5be49099b97ec3ff456145963746f` and Orange branch
`exp/gop-split-a16` at `6f6ea9e8f782d71d7f6f0b18a969a286ede9db95`. The representative
recording identifies Orange producer commit
`63d6b3538ec74407a6712cf0f82ca42ba71c1e36`; its Citrus producer commit is
unrecoverable from persisted metadata.
