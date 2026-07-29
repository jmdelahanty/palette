# Shared Zarr Logical Schema Catalog

Status: incremental foundation

## Purpose

Give Palette writers, publication validation, benchmarks, training exports, and
Crimson one versioned source of truth for array identity. Physical storage
profiles remain separate: logical schema says what an array means; storage
policy says how that exact logical array is encoded into chunks and shards.

## Current Inventory

The deterministic census in
`fisheye.diagnostics.zarr_storage_census` scans logical declarations and
physical writer sites independently. Its generated review artifacts are:

- `docs/diagnostics/zarr_array_schema_census.json`;
- `docs/diagnostics/zarr_production_writer_census.json`;
- `docs/diagnostics/zarr_storage_census_summary.md`.

The 2026-07-23 baseline found:

- 450 `ArraySpec` declarations;
- 270 unique leaf array names;
- 343 unique `(name, dtype, shape_template)` combinations;
- 43 leaf names with more than one dtype or shape signature;
- 1,558 physical writer/caller sites, including 660 direct Zarr API sites,
  897 statically resolved wrapper calls, and one array created by manual Zarr
  metadata plus encoded chunk objects;
- 487 statically resolved writer leaf names that have no `ArraySpec`
  declaration.

The largest repeats are structural fields: `frame_indices` appears 21 times,
`frame_counts` 10 times, `instance_key` and `source_refined_row_ids` 11 times
each, and `source_detect_row_index` 10 times.

This does not mean Palette has 450 unrelated scientific measurements. It means
stage composition currently repeats many logical declarations, while a subset
of leaf names has genuinely context-dependent meaning. A leaf name is therefore
not a sufficient schema identity.

`ArraySpec` itself remains useful validation evidence, but it is not the target
contract. It has a leaf name, free-form dtype string, symbolic shape, required
flag, and description. It has no canonical ID/version, exact cross-language
dtype, axes/units, physical storage policy, lifecycle, consumer, or writer
binding. Many analysis tuples also have no concrete path binding. The census
therefore never treats a declaration as proof that an array is written.

## Target Model

### Logical array contract

Each reusable semantic array family has a stable ID and version:

```text
palette.array.keypoints_img@1
palette.array.frame_counts@1
palette.array.subject_masks_roi_dense@1
```

Its contract owns:

- exact dtype/representation;
- rank, axis names, and fixed/symbolic shape constraints;
- scientific description;
- units and coordinate space;
- fill/null semantics when defined.

The contract does not own chunk sizes, shard sizes, codecs, lifecycle, or
benchmark results.

### Stage binding

A stage schema binds concrete paths to logical contracts:

```json
{
  "path": "keypoints_runs/<run>/keypoints_img",
  "contract_id": "palette.array.keypoints_img",
  "contract_version": 1,
  "required": true
}
```

Stage-specific arrays may keep their own contract. Shared lineage, coordinate,
mask, status, and index families are referenced instead of redeclared.

### Storage plan

The logical contract creates a concrete `ArrayIntent` after shape validation.
The resulting `StoragePlan` carries the logical schema ID/version, exact dtype,
shape, access pattern, and write mode alongside chunks, shards, codec profile,
and object estimates.

### Benchmark case

A benchmark case references:

- logical schema ID/version;
- resolved storage-policy/profile version;
- exact workload ID;
- source-data and environment identity.

Performance results remain separate evidence linked to the schema and profile.
Changing a timing result does not version the scientific schema; changing dtype,
axes, units, or coordinate meaning does.

## Exact Dtypes And Compatibility

Current schemas sometimes allow unions such as `float16/float32/uint8` or
`int16/int32`. Those are useful legacy compatibility declarations but are not a
current canonical contract.

Rules for new schema versions:

- one exact logical dtype per contract version;
- a dtype change requires a new schema version or a new representation ID;
- writers cast and validate before publication;
- Crimson selects a versioned adapter once;
- legacy alternatives are handled by explicit compatibility adapters, not
  repeated typed probes;
- benchmarks fail before timing when source, plan, and decoded destination
  dtypes disagree.

### How To Choose A Dtype

Choose a dtype from the scientific and operational contract rather than from
the values observed in one archive. For each array:

1. define its legal range, units, and signedness;
2. define the maximum acceptable quantization error in meaningful units such as
   pixels, degrees, seconds, or confidence;
3. account for nulls, missing-value sentinels, and out-of-range behavior;
4. identify whether errors can accumulate through interpolation, geometry,
   aggregation, or model training;
5. verify that Palette, Zarr/TensorStore, NumPy, and Crimson support the exact
   representation without repeated probing or implicit casts;
6. compare compressed bytes, decode cost, and working-memory cost on
   representative data;
7. require behavioral checks for thresholding, ranking, joins, cropping, and
   other discrete decisions that could change after quantization.

Storage dtype and compute dtype may differ, but that is a distinct versioned
representation. For example, a future fixed-point integer coordinate could be
decoded to `float32` for computation. Its scale, offset, range, rounding, and
overflow behavior would belong to the logical contract rather than being an
undocumented reader convention.

### Detection Geometry Decision

The first canonical detection contract uses exact `float32` for continuous
bounding-box and center geometry:

- `bbox_norm_coords`;
- `bbox_img_xyxy`;
- `centers_img_xy`.

For a normalized coordinate over a `4512`-pixel image, representative spacing
near the upper end of `[0, 1]` is:

| Representation | Approximate pixel spacing | Approximate maximum rounding error |
| --- | ---: | ---: |
| `float32` | `4512 * 2^-24 = 0.000269 px` | `0.000135 px` |
| `float16` | `4512 * 2^-11 = 2.203 px` | `1.102 px` |
| normalized `uint16` | `4512 / 65535 = 0.06885 px` | `0.03443 px` |

Floating-point spacing varies with magnitude; this table uses the largest
adjacent interval below `1.0` for normalized values. Integer quantization has
uniform spacing but requires an explicit encoding contract.

`float32` is intentionally conservative: it is comfortably below meaningful
image precision, directly supported by Palette and Crimson, and avoids adding a
quantization adapter while the shared schemas are still being established.
Current canonical-detection writers and archives that carry `float64` geometry
are an explicit transition/legacy representation; they are not evidence that
the new canonical contract should remain `float64`.

Experiments with `float16`, normalized `uint16`, or fixed-point `uint16` are
deferred until the canonical storage specifications and their consumers are
complete. Any later adoption requires a new schema version or representation
ID, quantified reconstruction error, and proof that downstream selection,
threshold, IoU, crop, review, and training behavior remains acceptable. This
deferral applies to detection geometry only; other scientific arrays receive
their own range and error-budget decisions.

## Metadata And Crimson

The archive-level schema/capability manifest lists concrete path bindings and
logical contract versions. Consolidated Zarr metadata supplies the actual array
metadata—shape, data type, chunk grid, and codecs—in one root view.

Crimson should:

1. open the published consolidated root metadata;
2. read the schema/capability manifest;
3. resolve supported logical contract versions;
4. validate expected dtype/rank against actual metadata once;
5. construct typed readers without probing multiple alternatives.

Consolidation reduces metadata requests. The manifest supplies semantic
authority; consolidated metadata alone does not decide which dtype is intended.

## Initial Foundation

`fisheye.shared.zarr.array_contracts` now defines exact contracts for:

- `frame_counts`;
- `frame_offsets`;
- canonical/refined detections and their lineage;
- geometry-only crops;
- legacy keypoint coordinate surfaces plus exact raw/refined keypoint-v2
  snapshots;
- immutable source-bound `keypoint_quality_runs` metric, flag, and proposal
  arrays;
- body-frame origin, axes, validity, and derived heading kept outside keypoint
  snapshots;
- dense subject masks in ROI space;
- flat contour `points_xy`.

The catalogs provide reusable identity, exact dtype validation, axis
constraints, JSON manifest export, and direct creation of storage-planner
intents. Logical contract coverage does not by itself activate a production
writer or selector.

## Implementation Priority

The phase-gated execution plan is
[`canonical_detection_storage_implementation_checklist.md`](canonical_detection_storage_implementation_checklist.md).

The first implementation wave targets current and future-facing authorities:

- canonical raw detections in `detect_runs`;
- immutable detection-quality snapshots in `detect_quality_runs`;
- current refined-detection authoring, snapshot, and training surfaces;
- common frame, observation-identity, lineage, status, and validity contracts
  reused by those stages;
- end-to-end writer, read, and publication benchmarks for each migrated
  canonical surface.

`detection_artifact_runs` is a deferred compatibility and diagnostic surface.
Its classification is:

- publication role: quarantined evidence;
- authority: noncanonical and explicitly unbound;
- lifecycle: immutable after construction and never selector-eligible;
- row identity: run-local and noncanonical;
- physical disposition: shard it when retained;
- implementation priority: do not migrate its writers or add dedicated
  benchmarks unless a supported future consumer or canonical binding path is
  approved.

The inventory continues to record its current arrays and conflicts so the
family is not accidentally mistaken for `detect_runs`. It is not a first-wave
schema, storage-policy, or benchmark target.

## Migration Checklist

### Catalog

- [x] Census all `ArraySpec` occurrences and group conflicting signatures.
- [x] Census direct, wrapped, training, publication/compaction,
      migration/legacy, derived-cache, and manual-metadata writer surfaces.
- [x] Preserve unresolved paths, dtypes, shapes, consumers, access patterns,
      and lifecycles explicitly for review.
- [x] Add exact dtype and logical array contract types.
- [x] Add stable ID/version lookup and JSON manifest records.
- [x] Carry schema identity from contract to storage plan.
- [ ] Define fill/null semantics as typed fields.
- [ ] Add canonical contracts for common lineage and validity/status families.
- [ ] Add canonical encoded-string/reason contracts.

### Stage schemas

- [x] Classify `detection_artifact_runs` as deferred, quarantined evidence rather
      than a canonical detection stage.
- [ ] Add contract references to `ArraySpec` or replace it with stage bindings.
- [x] Define logical canonical-detection and common frame/lineage contracts.
- [x] Define logical keypoint-v2, keypoint-quality-v1, and body-frame-v1 stage
      contracts with heading excluded from keypoint snapshots.
- [ ] Migrate keypoint contracts through their current authoring, publication,
      and training surfaces.
- [ ] Separate compatibility unions from current canonical declarations.
- [ ] Detect conflicting bindings for the same concrete path in CI.
- [ ] Export a complete stage-schema manifest.

### Writers and publication

- [ ] Require exact contract validation before array creation.
- [ ] Require exact contract validation before run completion.
- [ ] Publish concrete path bindings at the archive root.
- [ ] Compare manifest expectations with consolidated actual metadata.
- [ ] Reject selector publication on schema or dtype mismatch.

### Crimson

- [ ] Load schema bindings from the root manifest.
- [ ] Resolve one supported adapter per contract version.
- [ ] Remove repeated typed probing for current schema versions.
- [ ] Retain explicit legacy adapters only where historical archives require
      them.

### Benchmarks

- [ ] Require logical schema ID/version in the common benchmark envelope.
- [ ] Validate exact dtype and shape before starting timers.
- [ ] Record schema, storage profile, workload, data, and environment identity.
- [ ] Compare consolidated and unconsolidated metadata opening separately.
