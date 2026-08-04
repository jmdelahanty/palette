# Tail Trace Samples Export Contract V1

Date: 2026-08-04

Status: exact logical and 52-field Arrow schema, source binder, bounded
multipart publisher, decoded validator, CLI, and workflow/LSF adapter are
implemented. Short- and full-duration performance evidence remains. This
decision activates no workflow default, selector, registry authority, physical
profile, or canonical-data change.

## Decision

`tail_trace_samples` is an immutable, long-form Parquet query product with one
row per tail-kinematics observation and normalized tail-axis sample. It is not
the recording-local geometry authority. The selected `tail_kinematics`, source
`subject_shape`, and `track_kinematics` Zarr publications remain the exact
scientific, identity, and rollback authorities.

V1 deliberately does not use one variable- or fixed-list column per frame.
Tail sample cardinality is run-specific, whereas a primitive long-form table:

- preserves an explicit normalized `s` coordinate across runs;
- supports projection and predicate pushdown by tail position;
- avoids making run-local list width part of the cross-recording Arrow type;
- matches the accepted
  `recording x track x observation x normalized-tail-position` analysis grain;
  and
- can be partitioned into bounded immutable parts without decoding every
  profile for a narrow spatial query.

The cost is repeated row identity and a potentially large full-duration table.
This is therefore an explicit, selector-ineligible query product, never a
default compact export. The publisher must use bounded source-row windows and
multiple manifest-selected parts when representative-scale evidence requires
them.

## Identity And Tracking

The exact primary key is:

```text
(recording_id,
 source_tail_kinematics_run,
 source_tail_row_index,
 tail_sample_index)
```

`source_tail_row_index` preserves the canonical observation order, allowing a
bounded publisher to remain streaming even when rows from multiple tracks are
interleaved. Every row also carries exact `instance_key`, crop-row, camera-frame,
and `track_id` identities.

`instance_key` is an observation/edit-lineage identity, not an animal or track
identity. The publisher must join every tail observation to exactly one
selected track through `instance_key`; missing or duplicate mappings fail the
publication rather than producing track zero or a nullable ordinal. The core
workflow must consequently add `track_kinematics` as an explicit dependency of
`tail_traces`.

## Spatial Representation

The source tail run provides:

- normalized tail position `tail_angle_sample_s`;
- source-camera `tail_angle_sample_xy`;
- tangent angle relative to the caudal axis, positive anatomical-left;
- source-camera curvature in `px^-1`; and
- source-camera lateral deflection in pixels.

The bound subject-shape publication provides the anatomical body frame and the
per-row `tail_segment_arclength_px` reference length. For every sample:

```text
delta_xy = source_camera_xy - tail_base_xy
caudal_axis = -forward_axis
body_longitudinal_fraction = dot(delta_xy, caudal_axis) / reference_length_px
body_lateral_fraction = dot(delta_xy, left_axis) / reference_length_px
body_curvature_dimensionless = d(unwrapped_tangent_angle_rad) / d(normalized_s)
```

The longitudinal axis is therefore positive from tail base toward tail tip;
the lateral axis is positive anatomical-left. `body_curvature_dimensionless`
is the body-frame signed `curvature × reference_length` representation derived
from the authoritative tangent-angle surface. The source-camera point,
curvature, and pixel lateral deflection remain diagnostic columns; they are not
the cross-fish comparison representation.

The projection contract must persist the reference-length kind, handedness,
axis, angle, unwrap/derivative, invalid-value, and sampling conventions. A new
normalization or angle convention requires a new projection/schema version.

## Validity

All 52 fields are non-null. Invalid floating payloads use IEEE NaN and are
interpreted through explicit booleans and a closed uint16 reason registry:

- `0`: valid;
- `1`: source tail row invalid;
- `2`: reference length/body frame invalid;
- `3`: derived geometry non-finite.

`source_failure_reason` preserves the decoded source row reason as a
dictionary-encoded string. Valid samples require the tail row, finite positive
reference length, finite body axes/coordinates, and finite derived values.

## Source And Publication Gate

The publisher fails closed unless it can prove and recheck:

- explicit completed, selector-eligible tail-kinematics and track-kinematics
  selections;
- the exact tail coordinate-publication and array-schema manifests;
- the exact source subject-shape coordinate publication named by the tail run;
- complete, unique instance-key-to-track membership;
- identical frame, crop-row, and instance identities across bound sources;
- exact static sample axis, body-frame record, reference-length surface, and
  source dtypes; and
- unchanged source manifests and completion snapshots before visibility.

Construction occurs only on a non-overlapping node-local scratch root. The
publisher hashes every selected decoded source and output column, copies into a
hidden immutable generation, verifies the copy, and changes visibility only by
the existing manifest-exclusive compare-and-swap boundary. Publication remains
unindexed and selector-ineligible until short- and full-duration gates pass.

## Implementation Checklist

- [x] Freeze the long-form grain and exact primary key.
- [x] Freeze the ordered 52-field Arrow schema and source dtypes.
- [x] Register the table as a dedicated non-default streaming trace product.
- [x] Freeze body-frame normalization and dimensionless curvature semantics.
- [x] Add fail-closed tail, subject-shape, and track source binders.
- [x] Add deterministic instance-key-to-track join validation.
- [x] Add bounded source-window projection and multi-part Parquet writing.
- [x] Add full decoded-payload and physical Parquet validation.
- [x] Add source-change, payload-tamper, interrupted-replacement, and recovery
      tests.
- [x] Add CLI and workflow/LSF execution; include `track_kinematics` in the
      `tail_traces` dependency closure.
- [ ] Benchmark writer, copy, validation, narrow-tail-position reads,
      frame-window reads, complete scans, peak RSS, and object count at short
      and full duration.

## Validation Evidence

The initial logical/Arrow contract matrix passed 175 tests with ten expected
warnings from explicit legacy completion-compatibility fixtures. The executable
publisher matrix adds exact body-frame math, invalid-row semantics, unique
track joins, bounded multi-part writing, batch-independent decoded digests,
constant-column tampering, recomputed-digest nested source-declaration
tampering, and failed-replacement preservation. The final combined
workflow/Arrow/LSF matrix passed 233/233 with ten expected warnings from
explicit legacy completion-compatibility fixtures. The broader four-export
and atomic-publication regression matrix passed 113/113. Ruff, Python
compilation, shell syntax, and `git diff --check` pass.
