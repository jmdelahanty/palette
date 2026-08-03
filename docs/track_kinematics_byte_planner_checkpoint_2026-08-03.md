# Track-Kinematics Byte-Planner Checkpoint — 2026-08-03

Status: exact logical inventory frozen; byte-planned writer adoption blocked and
not enabled.

This checkpoint deliberately leaves the established track-kinematics writer,
activation, registry, and physical layout unchanged. It adds no production
profile, writes no selector, changes no `latest`/`latest_complete` pointer, and
does not make a candidate selector-eligible.

## Ground Truth

The maintained direct writer creates one run-level `track_ids` array, an
optional all-or-none `track_arena_ids` array, and one closed motion surface per
track. The existing full-motion validator defines the authoritative path
vocabulary in
`fisheye.analysis.track_kinematics._expected_motion_track_surface_paths`.

The newly executable census in
`fisheye.analysis.track_kinematics_schema` matches that vocabulary exactly:

| Surface | Arrays | Presence |
|---|---:|---|
| Per-track core motion | 69 | Required |
| Per-track physical peers | 35 | Optional all-or-none bundle |
| Run track inventory | 1 | Required |
| Run arena inventory | 1 | Optional all-or-none bundle |

The per-track physical bundle contains the exact millimetre peer for every
pixel array that has a controlled physical mapping. A physical run therefore
has 104 maintained arrays per track; a nonphysical run has 69.

The schema freezes complete-record access units. Sample-domain arrays are
windowed along the track-sample axis and never split their trailing two-value
identity/XY record. The four small second-domain summaries are eager. Every
array is an immutable-snapshot output. These classifications describe the
future planning input but do not claim adoption.

## Exact Dtypes And Fill Semantics

- `positions_px` and `positions_mm` remain exact `float64[N,2]`. This preserves
  the current source-coordinate authority and the maintained test requiring
  dtype-preserving position publication. This checkpoint does not silently
  narrow them to float32 merely because newer crop-v2 centers use float32.
- Motion, heading, path-distance, speed, acceleration, and per-second payloads
  are `float32`, except the position pair above.
- `track_ids`, `track_arena_ids`, and `delta_frames` are `int32`.
- Camera/source/frame/row/second indexes and `track_sample_key[N,2]` are
  `int64`.
- Detection source is `int8`; sample and transition reason codes are `int16`;
  validity arrays are `bool`.
- Invalid or unavailable floating motion uses NaN. Mandatory identity/time/code
  arrays use zero physical fill; arena identity uses `-1`; validity uses false.
- `heading_per_second_resultant` uses zero for no directional coherence.
- Nullable source-observation lineage uses the exact structured null record
  `valid=false, instance_key=0`.

The two required structured lineage arrays are exact current authorities:

1. `source_frame_interpolation`, 24 bytes/row:
   `left_source_frame_index:int64@0`,
   `right_source_frame_index:int64@8`, `right_weight:float64@16`.
2. `source_instance_key`, 9 bytes/row:
   `valid:bool@0`, `instance_key:uint64@1`.

The schema records field order, dtype, byte offset, and total itemsize rather
than reducing these records to opaque byte strings.

## Deliberately Excluded Compatibility Surfaces

The legacy `tracks/id_<N>/swim_bouts/**` mirror is not part of canonical track
motion authority. Its fields are inferred from an external structured bout
table, so admitting it would make the candidate inventory open-ended. New
consumers already read `analysis/swim_bout_runs` directly.

The run-root chaser auxiliary arrays (`camera_frame_ids`,
`stimulus_frame_nums`, `timestamp_ns`, `trial_state`, optional metadata/angle
fields) are likewise sealed legacy auxiliaries rather than public track-motion
coordinates. They are explicitly listed as excluded, not silently discovered
from a live run.

## Blocking Factory Mismatch

The shared byte planner is not changed by this checkpoint. The blocker is the
current logical-dtype/factory boundary for the two structured arrays:

1. `DTypeContract` stores one NumPy dtype string. NumPy cannot reconstruct
   either current structured dtype from `str(dtype)`.
2. `StoragePlan.logical_dtype` uses that same string identity, and the shared
   array factory calls `np.dtype(plan.logical_dtype)` before creation.
3. Zarr v3 represents structured `data_type` as an extension object and its
   structured fill as an encoded value, while the shared metadata comparator
   currently expects a scalar dtype string and ordinary fill value.

Using a track-local creation bypass would produce a nominal candidate that is
not governed by the shared factory. Reinterpreting the structures as opaque
`V24`/`V9` bytes would discard field semantics. Both options are rejected.

Therefore the exact simple declarations remain marked
`byte_planner_adopted=false`, and conversion of either structured declaration
to `AnalysisArrayDeclaration` raises a dedicated fail-closed error. No writer
candidate exists at this checkpoint.

## Proposed Versioned Resolution

The clean future schema is a versioned flattened lineage bundle:

- `source_frame_interpolation/left_source_frame_index: int64[N]`
- `source_frame_interpolation/right_source_frame_index: int64[N]`
- `source_frame_interpolation/right_weight: float64[N]`
- `source_instance_key/valid: bool[N]`
- `source_instance_key/value: uint64[N]`

That representation preserves every bit and semantic field while making each
array independently describable by `AnalysisArrayDeclaration`, plannable by
bytes, and creatable/validatable through the shared factory. It requires a new
track-lineage/run schema version plus reader, coordinate-binding, manifest, and
payload-validator changes; it must not be smuggled into the v1 path as a
physical-only rewrite.

The float64 position authority is a separate decision. A later version may
standardize new source-camera position publications on float32, but only after
the coordinate catalog, track input authority, numerical equivalence policy,
and all consumers agree. Until then, track publication must preserve float64.

## Implementation Checklist

- [x] Freeze the 69-array required per-track core vocabulary.
- [x] Freeze the 35-array all-or-none physical peer bundle.
- [x] Freeze required run track identity and optional arena identity.
- [x] Record exact dtypes, shapes, axes, access classes, authority roles, fills,
      null semantics, units, and coordinate spaces.
- [x] Record the two structured dtype field layouts and item sizes exactly.
- [x] Prove the census equals the maintained writer/full-motion vocabulary.
- [x] Exclude dynamic swim-bout mirrors and legacy chaser auxiliaries explicitly.
- [x] Fail closed when structured declarations are converted to the current
      shared `AnalysisArrayDeclaration` boundary.
- [ ] Approve a versioned flattened-lineage contract.
- [ ] Update track readers, coordinate binding, publication manifest, and
      payload validation for that version.
- [ ] Decide whether future position authority remains float64 or transitions
      to a separately versioned float32 coordinate contract.
- [ ] Build the selector-ineligible byte-planned writer only after every exact
      declaration is representable by the shared factory.
- [ ] Persist and recompute one complete storage-plan receipt.
- [ ] Validate direct and consolidated metadata equivalence.
- [ ] Benchmark publication and consumer reads before promotion.

## Safety Result

This worktree contains schema, tests, and this document only. It does not edit
the shared planner, shared profiles, storage catalog, registry, selectors, or
production archives. The default track writer remains the only writer and its
behavior is unchanged.
