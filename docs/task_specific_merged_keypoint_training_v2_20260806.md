# Task-specific merged keypoint training v2 — 2026-08-06

Status: **IMPLEMENTED; FIRST SELECTOR-INELIGIBLE CANARY PUBLISHED**

## Decision

Merged training artifacts are task-specific immutable products, not mutable
unions of every historical training Zarr. A source enters a merge only after
its review state, intended use, skeleton semantics, pixel authority, crop
lineage, and keypoint row gate are explicit. The merge retains exact per-row
source lineage and keeps registry activation deferred until the artifact has
been trained and evaluated.

The initial historical cohort contains seven safely preflighted sources: four
approved Sickyfish datasets and three approved Sleepyfish datasets. This is
not a repository-wide count of usable keypoint datasets. The reviewed Batman
artifact is the eighth input to the first task-specific merge. Other archives
remain excluded until their review/quality state, skeleton identity, pixel
contract, or legacy crop lineage is resolved.

## Frozen v2 behavior

- Build on bounded node-local scratch, validate direct metadata and the exact
  storage plan, consolidate as the final immutable visibility step, copy to a
  hidden sibling, compare physical inventories, and publish by atomic rename.
- Keep the published artifact immutable, selector-ineligible, and absent from
  the registry until a later explicit activation decision.
- Normalize every output ROI to one explicit canvas by centered zero padding.
  Resizing is forbidden. Keypoint XY coordinates are translated by the exact
  padding offset and the transform is persisted per source.
- Split by source dataset so rows from one recording/source cannot leak across
  train and validation partitions.
- Persist `keypoints_roi` as `float32`. Strict mode rejects mixed source
  dtypes. The opt-in `float32_checked` compatibility policy accepts only
  float32/float64, verifies finite-value preservation and a magnitude-scaled
  conversion bound, and records the measured maximum error per source.
- Derive chunks and shards from logical bytes and access patterns through the
  shared storage planner. The 512x512 `uint8` ROI surface uses 1 MiB inner
  chunks (four samples) inside four large indexed shards for this canary;
  small fixed-width arrays collapse to one object when appropriate.
- Preserve source dataset, source Zarr, frame, ROI row, refined row, and raw
  detection-row lineage. Masks are not a dependency of a keypoint-only merge.

## First canary

Published under:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/keypoint_merged_v2/batman_plus_approved_traditional_v2_v002/`

The authoritative persisted source-manifest composition digest is
`65ae8ac82545ee1879bf43c7e39759ec6288918c19ba71d7bd19e642cf6067d5`.

An earlier selector-ineligible v001 integration artifact is retained as
superseded benchmark evidence. Its publication receipt exposed the temporary
validation path; v002 replaces that diagnostic field with the immutable final
target path. Neither artifact is registered or production-selected.

Results:

- 8 sources and 1,682 reviewed/usable rows;
- 1,247 training rows and 435 validation rows, grouped by source dataset;
- exact `pose_skel_traditional_v2` semantics with five landmarks;
- 512x512 grayscale `uint8` pixels, with Batman padded from 348x348 and no
  source resized;
- canonical `float32` keypoints; historical float64 conversion errors were at
  most approximately `1.526e-5` pixels and Batman required no conversion;
- 45 physical files and approximately 255 MiB stored;
- complete immutable publication, consolidated metadata, deferred registry
  activation, and `stage_selector_eligible=false`.

The canary is implementation and training-input evidence. It does not promote
a model, activate a production selector, or alter any source artifact.

## Implementation checklist

- [x] Publish a keypoint-only reviewed source artifact independent of mask
  review state.
- [x] Compose approved historical manifests with the reviewed Batman source.
- [x] Enforce one skeleton and label ordering.
- [x] Implement explicit pad-without-resize ROI transforms.
- [x] Implement source-dataset-grouped splits.
- [x] Add exact merged-v2 array schemas and byte-derived storage plans.
- [x] Add checked float64-to-float32 compatibility with per-source receipts.
- [x] Build, validate, consolidate, and atomically publish the first canary.
- [ ] Visually sample the padded Batman rows and representative historical
  rows in the training viewer.
- [ ] Train and evaluate a candidate model using the frozen split.
- [ ] Register this exact artifact only if the model/lineage review passes.
- [x] Census the complete approved historical five-point corpus and the
  reviewed Batman candidate; see
  `docs/keypoint_training_source_census_2026-08-06.md`.
- [x] Add dual frame-domain lineage and subject/cohort-grouped splits before
  rematerializing the complete 61-source cohort; implemented as the immutable
  v3 successor in `docs/task_specific_merged_keypoint_training_v3_20260807.md`.
