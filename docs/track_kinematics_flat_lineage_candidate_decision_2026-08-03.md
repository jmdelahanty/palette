# Track-Kinematics Flat-Lineage Candidate Decision — 2026-08-03

Status: implemented as an explicit, selector-ineligible physical candidate;
not promoted and not used by the production writer or selectors.

## Decision

The maintained track-kinematics v1 authority remains unchanged. In particular,
`positions_px` and the optional all-or-none `positions_mm` peer remain exact
`float64[N,2]` arrays. The candidate does not narrow them to float32.

The two v1 NumPy structured lineage arrays are not suitable as a future public
cross-language Zarr contract. Zarr Python warns that both structured Zarr-v3
data types are unstable, and the shared Palette planner/factory intentionally
cannot reconstruct them from its exact primitive dtype identity. The v2
candidate therefore uses these primitive arrays:

| v1 record | v2 primitive path | dtype | null rule |
|---|---|---:|---|
| `source_frame_interpolation.left_source_frame_index` | `source_frame_interpolation/left_source_frame_index` | `int64[N]` | none |
| `source_frame_interpolation.right_source_frame_index` | `source_frame_interpolation/right_source_frame_index` | `int64[N]` | none |
| `source_frame_interpolation.right_weight` | `source_frame_interpolation/right_weight` | `float64[N]` | none |
| `source_instance_key.valid` | `source_instance_key/valid` | `bool[N]` | false means absent |
| `source_instance_key.instance_key` | `source_instance_key/value` | `uint64[N]` | must be zero when `valid=false` |

This is a logical schema version, not a physical-only rename. The candidate
manifest identifies v2 explicitly and records v1 as an explicit compatibility
source. `load_track_lineage_records(..., lineage_schema_version=1|2)` is the
only compatibility helper added here: callers choose a version and the v2
fields are reconstructed bit-for-bit into the established v1 in-memory record.
There is no dtype probing or silent fallback.

## Physical Candidate

`fisheye.analysis.track_kinematics_storage` builds one closed declaration set
for every track in the run:

- 72 primitive core arrays per track (`69 - 2 + 5`);
- the same 35-array all-or-none physical peer bundle;
- `track_ids` and optional `track_arena_ids` at run scope; and
- no mirrored `swim_bouts` or historical chaser auxiliary arrays.

Every declaration is passed to the shared byte planner with the explicit
`published_http_v1` profile and created only through the shared array factory.
Consequently Zarr format 3, the `zstd_fast_v1` codec chain, byte-derived inner
chunks, and data-size-derived sharding are receipt-bound. Small fixtures may
legitimately remain one/few unsharded objects; full-duration windowed arrays
derive shards from their logical bytes rather than from one hard-coded frame
count. Every track has a unique symbolic sample/second dimension in the
run-wide receipt, so different track lengths cannot accidentally constrain one
another.

The writer streams bounded row blocks from the v1 authority to node-local
scratch. It does not load the complete motion run into memory, does not use
Dask, and has one serial owner for every complete physical array/chunk/shard.
The published candidate is imported through the common atomic run-group
publisher.

## Publication And Failure Safety

The materializer accepts one explicit complete, selector-eligible v1 source
run and one new candidate name. It rejects:

- the same source and candidate name;
- an existing target;
- scratch inside the authoritative archive;
- the archive inside scratch; and
- symlink aliases of those containment cases.

The source is opened from the published consolidated metadata generation.
Computation/rematerialization occurs only in the fresh scratch tree. Before
publication, Palette validates the closed array inventory, primitive dtypes,
the recomputed storage-plan receipt, exact decoded hashes against every v1
field, the nullable-instance rule, completion/ineligibility state, and direct
versus consolidated metadata.

Atomic import leaves all `latest`, `latest_complete`, and `latest_offline`
pointers unchanged. It consolidates the authoritative root only after the
candidate is complete and selector-ineligible, then compares direct and
consolidated attrs/array declarations. The common publisher retains an
owner-bound failed tombstone on a post-rename failure; this family supplies a
repair callback that reconsolidates the failed visibility state.

No registry row, production profile, selector, training artifact, or existing
run is changed.

## Invocation

Planning is read-only unless `--apply` is supplied:

```bash
scripts/py -m fisheye.analysis_workflows.materializers.track_kinematics_candidate \
  RECORDING_analysis.zarr \
  --source-run SOURCE_V1_RUN \
  --run-name CANDIDATE_V2_RUN \
  --scratch-root /scratch/$USER/$LSB_JOBID/track-flat-candidate \
  --profile published_http_v1 \
  --apply
```

The profile choice is deliberately closed to `published_http_v1` in this
checkpoint. A different physical experiment needs a new explicit profile and
benchmark decision rather than an unrecorded CLI constant.

## Implementation Checklist

- [x] Preserve exact float64 position authority.
- [x] Replace both structured records with five exact primitive arrays in v2.
- [x] Retain explicit v1 read/reconstruction compatibility.
- [x] Freeze a closed run-wide primitive array declaration inventory.
- [x] Use the shared byte planner and Zarr-v3 array factory for every array.
- [x] Bind one complete storage-plan/profile/codec/object-estimate receipt.
- [x] Stream v1 data through node-local scratch.
- [x] Prove exact decoded equality field by field.
- [x] Validate direct and consolidated metadata locally and after publication.
- [x] Publish atomically without selectors, registry, or profile promotion.
- [x] Reject source/scratch/target aliasing and containment.
- [x] Supply failed-publication consolidated-visibility repair.
- [ ] Run one full-duration selector-ineligible publication canary.
- [ ] Have Crimson validate exact typed v2 opens and v1/v2 logical equality.
- [ ] Update production coordinate binding/read selection only under a later
      promoted v2 run contract; the candidate is not canonical input today.
- [ ] Benchmark object count, publication time, eager inventory reads,
      windowed sample reads, random seek, traversal transfer, and RSS.
- [ ] Define a separate promotion decision and production writer migration only
      if those gates pass.

## Compatibility Boundary

Historical v1 arrays remain readable only through the explicit v1 branch. A
v2 candidate must contain no structured array and must never be presented as a
v1 run. Conversely, this checkpoint does not mutate or replace a v1 run and
does not teach the production selector to choose v2. That separation is the
reason the physical experiment is safe to run before any promotion decision.
