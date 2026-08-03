# Tail-Kinematics Byte-Planner Candidate Adoption — 2026-08-03

Status: implemented as an explicit, unpromoted candidate. The legacy writer
and production publication path remain the default.

## Contract frozen in this checkpoint

`analysis/tail_kinematics_runs/<run>` now has a stage-owned exact array schema:

- 21 required arrays;
- one optional, all-or-none two-array
  `source_refined_subject_masks_revision` bundle;
- exact `uint64`, `int64`, `bool`, `uint8`, and `float32` dtypes;
- exact symbolic row, tail-sample, component, vector, and fixed-width UTF-8
  byte axes;
- explicit lineage, scientific-authority, and quality-diagnostic roles;
- explicit null/fill semantics and eager/windowed access classes; and
- immutable write lifecycle for every array.

The optional bundle is rejected when only `row_revision` or only
`row_revision_available` is present. Candidate mode also requires exact
`uint64 instance_key`, `int64 source_crop_row_ids`, and `int64
source_acquisition_frame_index` source lineage. Older logical snapshots remain
readable through the unchanged legacy path.

## Opt-in candidate behavior

The candidate is selected only with:

```text
--storage-profile published_http_v1
```

For every concrete array, it:

1. binds the frozen logical declaration to concrete shape and dtype facts;
2. derives inner chunks and outer shards from uncompressed bytes and access
   shape through `analysis_storage_planning`;
3. creates the array through the shared Zarr-v3 array factory;
4. persists the digest-bound `AnalysisStoragePlanReceipt`;
5. executably replans the receipt from the live array inventory;
6. validates actual chunk, shard, Zstd, CRC, fill, and array metadata;
7. consolidates the final immutable metadata generation; and
8. compares every candidate array's direct declaration with its consolidated
   declaration.

Candidate mode permits one serial writer only. `process_shards` is rejected
instead of assuming that a logical row partition owns every planner-derived
physical shard. This follows Palette's whole-physical-chunk/shard ownership
rule.

## Publication boundary

Candidate runs are complete and immutable but always remain
`stage_selector_eligible = false`. They do not update `latest` or
`latest_complete`, emit no serialized registry completion, and do not change
the default storage profile. The node-local materializer may atomically copy a
candidate into the archive, but its final callback only consolidates and
validates metadata; it never commits coordinate-selection activation.

The existing writer behavior remains unchanged when `--storage-profile` is
omitted, including legacy physical chunk/shard policy and normal activation.

## Subject-shape boundary discovered in this checkpoint

Subject shape is not yet safe to migrate in the same patch. Unlike tail
kinematics, its live array inventory is component-dependent and is assembled
across component, relation, body-frame, source-revision, and coordinate-binding
helpers. The writer also contains a direct array-creation bypass for one
surface. Although canonical materialization currently requires the full
component anchor set, no stage-owned `AnalysisArrayDeclaration` builder closes
the exact component-dependent inventory and its optional-bundle rules.

Before a subject-shape candidate writer is authorized:

- freeze the exact array set for each supported component-set profile;
- decide whether only the full canonical component set is maintained or
  whether multiple component-set schema variants are supported;
- close relation/body-frame presence rules as named all-or-none bundles;
- freeze exact lineage and component-revision dtypes;
- remove the remaining direct array-creation bypass; and
- add adversarial inventory tests before creating any byte-planned arrays.

Creating a permissive declaration set from whichever arrays happened to be
written would make the physical receipt self-consistent but would not freeze a
logical contract, so this checkpoint deliberately does not add subject-shape
candidate mode.

## Validation checklist

- [x] Exact 21-array core declaration set.
- [x] Exact two-array optional revision bundle.
- [x] Semantic physical fills.
- [x] Byte-derived plans preserve full trailing records.
- [x] Receipt digest and executable replanning.
- [x] Actual chunks/shards/codecs/fills validation.
- [x] Direct/consolidated array declaration equivalence.
- [x] Serial-only candidate write enforcement.
- [x] Complete but selector-ineligible lifecycle.
- [x] Parent selector non-mutation.
- [x] Legacy default behavior retained.
- [ ] Full-duration candidate benchmark and promotion decision.
- [ ] Subject-shape logical-schema closure and candidate adoption.
