# GoodBatBadBat Acquisition Crop-Stream Production Checklist

<!-- contract-meta
status: implementation-checklist
last_verified: 2026-08-15
target_profile: full_plus_crop
purpose: Productionize complete acquisition crop-stream import and hybrid crop-pixel consumption for the existing GoodBatBadBat full-plus-crop recordings.
-->

## Outcome

For every current GoodBatBadBat recording, make the acquisition crop stream a
complete, consumer-facing Zarr input while preserving the normal scientific
workflow:

```text
full-frame video
  -> offline raw detection
  -> detection quality
  -> reviewed refinement
  -> canonical crop geometry
  -> per-row pixel routing
       acquisition crop video when valid and sufficient
       full-frame recovery otherwise
  -> keypoints and subject masks
```

The acquisition crop video supplies high-fidelity pixels. It does not replace
offline detection authority, quality, refinement, reviewed dish geometry, or
explicit downstream source selection.

This document is an implementation checklist. Checked items describe code on
the implementation branch; they do not authorize selector changes, recording
mutation, production jobs, or crop-only retention. Required CI and a
selector-ineligible real-recording canary remain mandatory before deployment.

## Corpus snapshot

A read-only registry report generated on 2026-08-15 found:

| Fact | GoodBatBadBat state |
| --- | ---: |
| Active analysis recordings | 84 |
| Both `full` and `crop` streams inventoried | 84 |
| Crop MP4 exists | 84 |
| Crop metadata exists and parses as `ok` | 84 |
| Crop dimensions | 384 x 384 for all recordings |
| Offline detection surface present | 84 |
| Analysis Zarr with any `crop_runs` child | 1 |
| Usable acquisition crop rows | 14,075,542 |
| Blank/no-detection rows | 126,850 |
| Minimum usable fraction | 0.9164464939 |
| Median usable fraction | 0.9969930516 |
| Maximum usable fraction | 1.0 |

The snapshot report used the legacy `min_crop_size=348` readiness threshold.
Because every observed acquisition crop is exactly 384 x 384, this does not
change the inventory conclusion, but production planning must rerun the report
with a 384-pixel minimum.

The report's recommended action was
`build_analysis_acquisition_crop_run` for 83 recordings and
`run_analysis_keypoints_from_roi_provider` for one recording. That recommendation
describes the existing utility surface, not the desired final production policy.
The production path below first canonicalizes every raw ledger row and then
builds crop geometry from the offline refined rowset.

Representative canary:

```text
recording:
  2026-08-10T17-20-55Z_arena_1_goodbatbadbat

analysis Zarr:
  /groups/johnson/johnsonlab/jeremy/recordings/
  2026-08-10T17-20-55Z_arena_1_goodbatbadbat/zarr/
  2026-08-10T17-20-55Z_arena_1_goodbatbadbat_analysis.zarr
```

Its existing 348 x 348 geometry-only crop run is an immutable canary artifact,
not evidence that acquisition crop ingestion is already production-complete and
not the intended new zebrafish geometry. Do not overwrite or promote it as part
of this work. Publish a 384 x 384 successor under the new crop-policy identity.

## Current implementation boundary

### Present

- Normal recording import mirrors `recording_manifest.json` stream declarations
  under `analysis/acquisition_video_streams/streams/{full,crop}`.
- The mirror records paths, existence, file size, metadata row count, dimensions,
  frame clock, coordinate declarations, codec, recorder summary, and status.
- The registry has acquisition-video-stream rows and current-recording views.
- `build_analysis_acquisition_crop_run` can parse crop metadata and publish a
  geometry-only acquisition-backed crop run.
- `CropImageSource` can decode acquisition crop-video rows through
  `source_crop_video_frame_indices`.
- `build_hybrid_acquisition_offline_crop_run` can combine acquisition crop-video
  rows with full-frame recovered rows in a supplemental flat cache.
- Offline raw detection, geometry gating, quality, and refined-detection products
  exist for the GoodBatBadBat campaign.
- The implementation branch publishes
  `palette.acquisition_crop_stream_ledger.v1` as pointer-last immutable runs
  below `analysis/acquisition_video_streams/streams/crop/ledger_runs`.
- Normal recording import and the explicit acquisition-stream backfill command
  use the same complete-ledger writer.
- The registry distinguishes crop-media availability from
  `crop_stream_consumer_ready` ledger readiness.
- New zebrafish workflow defaults and production entry points use 384 x 384;
  persisted 348 x 348 artifacts remain unchanged.

### Missing or not production-complete

- The implementation branch is not deployed and has not passed required CI.
- A selector-ineligible disposable canary against the representative real
  GoodBatBadBat sources passed with 152,035 rows, 149,989 detected rows, 2,046
  blank rows, 25 arrays, four hashed sidecars, and validated consolidated
  visibility. The canonical analysis Zarr was not opened for writing.
- Existing recordings have not yet been backfilled with the complete ledger.
- Live-event sidecars are not yet canonicalized. Declared crop metadata,
  keyframe, summary, and status sidecars are SHA-256 bound; the large MP4 uses
  the existing documented `stat_v1` fingerprint.
- Full-frame bounds validation for crop origins still needs a recording-level
  native-extent authority where Orange's full-stream manifest omits dimensions.
- The current acquisition crop-run geometry is based on the selected live box;
  production GoodBatBadBat crop geometry should remain bound to the selected
  offline refined-detection rowset.
- Acquisition detection import remains a nonselector
  `detection_artifact_runs` compatibility surface; canonical promotion is not
  implemented.
- Whole-recording production keypoint orchestration still assumes a flat ROI
  cache rather than a generic acquisition/hybrid provider manifest.
- Registry readiness now expresses raw-stream canonicalization, but routing
  completeness and downstream provider consumption remain future stages.
- Crimson has not validated direct raw acquisition-stream reads or the intended
  acquisition/hybrid crop-run consumer chain.

## Fixed invariants

- Do not modify source MP4s, producer CSV/JSON/JSONL files, existing raw
  detections, reviewed refined detections, geometry selections, or immutable
  analysis runs.
- Keep all raw acquisition ledger rows, including blanks, no-detection outcomes,
  and failures.
- Keep `crop_xywh` separate from `detection_xywh`.
- Keep acquisition detections separate from offline detections.
- For current GoodBatBadBat science, offline reviewed/refined detections remain
  authoritative.
- Use the exact selected refined-detection rowset as the fish/bbox authority.
- Use `zebrafish_crop_384_v1` as the crop geometry policy for new GoodBatBadBat
  crop runs.
- Acquisition-backed rows use the complete recorded 384 x 384 crop and its
  recorded native full-frame origin; fallback rows use a 384 x 384 ROI generated
  from the selected refined detection.
- Use acquisition crop pixels only when an exact frame match and containment
  policy passes.
- Recover insufficient/missing acquisition rows from the full-frame video.
- Preserve one ordered crop row for every selected refined-detection instance,
  with exact `instance_key` coverage.
- Never infer pixel source from crop-stream availability.
- Never apply a presentation reflection or heuristic axis flip.
- Never add an undeclared tolerance to producer dish geometry.
- Publish new immutable generations; do not overwrite or average prior evidence.
- Use unconsolidated reads during mutation and consolidate only after complete
  publication and selector metadata validation.
- Do not activate production selectors until required CI and canary evidence are
  complete.

## Phase 0: Freeze schemas and policy identities

- [x] Define a versioned raw acquisition crop-stream schema under
  `analysis/acquisition_video_streams/streams/crop`.
- [x] Decide whether the first version extends the existing stream group in
  place or publishes immutable import generations with an explicit current
  pointer. Prefer immutable generations if repeat import can produce differing
  bytes or metadata.
- [x] Freeze names, dtypes, dimensions, null/sentinel rules, chunking, and
  coordinate semantics for the complete frame ledger.
- [x] Freeze the distinction between:
  - raw acquisition stream rows;
  - all live detector observations;
  - the one live detection selected by the crop controller;
  - derived analysis crop rows.
- [ ] Define a stable stream reference containing recording, camera, stream ID,
  frame clock, native full-frame extent, crop-video extent, source contract, and
  immutable import-generation digest.
- [ ] Define a source-media identity ladder:
  - producer/transfer content checksum when available;
  - a one-time whole-container byte hash when required and operationally
    acceptable;
  - a versioned bounded fingerprint only when explicitly permitted, labeled as
    weaker than a content hash.
- [x] Do not hash every decoded frame. The first implementation uses a full
  SHA-256 for crop metadata and the documented cheap `stat_v1` MP4 fingerprint.
  Benchmark sequential container hashing separately from decode before changing
  that campaign policy.
- [x] Define a sidecar identity policy. Crop metadata, keyframe, summary, status,
  and live-event sidecars are small enough to receive full content hashes.
- [x] Define stream import completion and validation schemas.
- [ ] Define a versioned crop-pixel routing policy and reason-code vocabulary.
- [ ] Define `zebrafish_crop_384_v1` as a species-aware 384 x 384 crop geometry
  profile rather than changing an unexplained global integer default.
- [x] Inventory and update all production-facing 348 defaults, including shared
  crop defaults, default configuration, geometry-review approval planning,
  readiness reporting, preflight, and crop publication entry points.
- [x] Keep persisted crop geometry separate from model tensor size. Reuse the
  existing reversible `ModelInputTransform` for identity or centered zero-padding
  to a larger submitted extent, and retain its declared preprocessing identity.
- [x] Preserve existing 348 x 348 runs unchanged and classify them through their
  original crop-policy identity.
- [ ] Freeze full-frame edge handling for 384 x 384 fallback crops, including
  whether the ROI origin is clamped or pixels outside the native extent are
  zero-filled. Record that decision in `zebrafish_crop_384_v1`.
- [ ] Define a versioned GoodBatBadBat crop-sufficiency policy. Do not invent
  pass thresholds merely to enable automation.
- [x] Record explicit non-goals for the first implementation:
  - no full-video deletion;
  - no live-detection promotion to scientific authority;
  - no pre-inference dish masking change;
  - no subject-mask campaign until keypoint provider parity is established.

## Phase 1: Canonical complete acquisition crop import

### Import arrays

- [x] Add row-aligned Zarr arrays for every crop-stream frame, including at
  least:

```text
source_recording_frame_ids
source_recording_frame_indices
source_crop_meta_row_indices
source_crop_video_frame_indices
source_crop_local_frame_ids
source_camera_frame_ids, when supplied
acquisition_timestamps, when supplied
acquisition_system_timestamps, when supplied
has_detection
blank_frame
source_crop_xywh
selected_live_detection_xywh
selected_live_detection_confidence
```

- [x] Preserve producer row order and an explicit source CSV row index.
- [x] Store crop placement in native full-frame `xywh` pixels.
- [x] Store the selected live detection as separate provenance, not canonical
  crop geometry and not a reviewed detection authority.
- [x] Preserve blank/no-detection rows instead of dropping them.
- [x] Attach array/group-level coordinate, index-base, unit, and sentinel semantics.
- [x] Use chunking suitable for Crimson/TensorStore frame-window reads and
  Palette vectorized scans; record the profile ID.

### Validation

- [x] Resolve the crop stream only from the exact recording manifest declaration.
  Do not scan for newest or convenient files in production import.
- [ ] Require exact recording, camera, frame clock, and native extent agreement.
  Recording/camera/frame-clock and crop-video extent are enforced; full-frame
  bounds remain pending where producer manifests omit full-stream dimensions.
- [x] Require declared crop metadata row count and crop-video frame count to
  agree, or publish an explicit failed/incomplete import without guessing.
- [x] Validate `recording_frame_id` monotonicity, uniqueness, and index-base
  contract.
- [x] Validate crop-video frame-index coverage and dropped-frame semantics.
- [x] Validate `has_detection`/`blank_frame` combinations.
- [ ] Validate finite, positive, in-bounds crop geometry for nonblank rows.
- [x] Validate selected live-detection geometry separately from crop geometry.
- [ ] Bind summary/status/keyframe declarations and their checksums.
- [ ] Record parser, schema, software, and configuration identity.
- [x] Publish complete frame-domain evidence and summary counts.
- [x] Make exact repeat import idempotent; make disagreement fail closed rather
  than overwrite.

### Publication

- [x] Stage each recording's new Zarr payload as an unselected immutable run.
- [x] Validate all arrays, attrs, source identities, and logical digests before
  publication.
- [x] Publish atomically for readers within one recording Zarr by selecting the
  immutable run only after payload completion.
- [x] Update direct selection/import metadata only after payload validation.
- [x] Consolidate the root as the final visibility step and validate that the
  consolidated generation includes the new stream state.
- [x] Add a focused backfill command for existing GoodBatBadBat recordings and
  make normal import perform the same operation for future recordings.
- [ ] Emit one durable per-recording result receipt suitable for campaign-level
  reconciliation.

## Phase 2: Complete live-detection evidence

- [ ] Inventory whether each GoodBatBadBat recording contains Orange
  `yolo_events.jsonl` or an equivalent complete live detector stream.
- [ ] Canonicalize complete live detector events separately from the selected
  crop-controller box.
- [ ] Preserve model/weights, preprocessing, thresholds, NMS, class mapping,
  software, and selection-policy identity.
- [ ] Bind every event to recording, camera, frame clock, native extent, and
  source pixel contract.
- [ ] Preserve frames with detector errors or no detections as explicit
  frame-domain outcomes, not fake instance rows.
- [ ] Retain `detection_artifact_runs` as an explicit compatibility/audit route.
- [ ] Design and test a separate canonical promotion boundary into
  `detect_runs` only after exact coordinate and row identity are supported.
- [ ] Keep any promoted acquisition detection run nonauthoritative for the
  current GoodBatBadBat workflow unless a later reviewed policy explicitly
  selects it.
- [ ] Extend realtime/offline comparison to consume the canonical Zarr
  acquisition arrays instead of reparsing Orange CSV/JSONL files.

## Phase 3: Canonical refined-detection crop routing

### Rowset

- [ ] Resolve one exact, complete, selector-eligible refined-detection run and
  its exact dish-gate/selection lineage.
- [ ] Build one canonical 384 x 384 crop row per selected refined-detection
  instance using `zebrafish_crop_384_v1`.
- [ ] Preserve the refined-detection `instance_key` as the crop observation key;
  do not mint a replacement identity from the live acquisition box.
- [ ] Require identical ordered `instance_key` coverage between the selected
  refined instances and the derived crop rowset.
- [ ] Preserve `source_refined_row_ids`, frame indices, acquisition-frame
  indices, full-frame ROI placement, bbox geometry, and crop-policy identity.

### Acquisition-video eligibility

For each canonical crop row:

- [ ] Resolve exactly one raw acquisition crop ledger row for the same recording
  frame.
- [ ] Require a decodable, nonblank crop-video frame with exact 384 x 384
  dimensions and valid full-frame placement.
- [ ] Treat the complete recorded 384 x 384 acquisition crop window as the
  candidate ROI; do not take a smaller subwindow.
- [ ] Test whether the selected refined fish bbox and the versioned required
  context margin are contained within that recorded crop window.
- [ ] When eligible, use the complete crop-video frame and record its exact
  native full-frame origin from `source_crop_xywh`.
- [ ] Do not use the selected live bbox as the canonical analysis bbox.
- [ ] Record full-precision containment margins and policy result.

### Full-frame fallback

- [ ] Route blank, missing, ambiguous, undecodable, or insufficient acquisition
  crops to a 384 x 384 full-frame recovery ROI generated from the canonical
  refined detection.
- [ ] Materialize only fallback rows into a node-local supplemental flat cache.
- [ ] Bind the cache manifest to the exact full-frame media identity, crop run,
  row IDs, crop policy, and software identity.
- [ ] Preserve explicit routing reasons, including at least:

```text
acquisition_crop_selected
blank_acquisition_crop
acquisition_no_detection
acquisition_crop_missing
acquisition_crop_decode_failed
canonical_roi_not_contained
frame_identity_mismatch
coordinate_or_extent_mismatch
full_frame_recovery_selected
unrecoverable
```

- [ ] Fail closed if any canonical row has no valid provider.

### Immutable routing artifact

- [ ] Publish the provider decision as part of the canonical crop run or as one
  exactly bound immutable provider manifest.
- [ ] Store source kind, raw stream row, crop-video frame, full-frame ROI origin,
  supplemental-cache row, and reason code per crop row.
- [ ] Bind the exact raw acquisition stream generation, refined detection run,
  full-frame source, crop policy, and geometry selection digests.
- [ ] Ensure provider policy can change by publishing a successor without
  rewriting raw acquisition arrays or refined detections.

## Phase 4: Production model-consumer integration

- [ ] Generalize the whole-recording keypoint planner from mandatory flat-cache
  input to an explicit ROI provider manifest.
- [ ] Inventory the exact pose-model input contract for every selected zebrafish
  keypoint/mask model.
- [ ] Feed native 384 x 384 input directly when model and stride contracts permit
  it; otherwise use the existing centered `pad_to_size` transform for a larger
  submitted extent and its existing inverse coordinate mapping.
- [ ] Record any later Ultralytics/network resize as the separate framework
  preprocessing stage already required by the pose model-input contract.
- [ ] Do not silently center-crop acquisition frames back to 348 x 348.
- [ ] Support provider blocks for:
  - complete 384 x 384 acquisition crop-video frames;
  - full-frame supplemental flat-cache rows.
- [ ] Group work by provider and decode source rather than switching decoders
  row by row.
- [ ] Merge model outputs back into canonical crop-run row order.
- [ ] Require exact `source_crop_row_ids` and `instance_key` coverage in raw
  keypoint output.
- [ ] Extend terminal receipts and finalizers to bind crop run, provider
  manifest, source media, rowset digest, model, and configuration.
- [ ] Make stale, partial, duplicated, reordered, or mismatched provider rows
  block publication.
- [ ] Record whether each output row used acquisition crop pixels or full-frame
  recovery.
- [ ] Preserve full-frame keypoint coordinates by projecting through canonical
  crop placement, not through the live detection box.
- [ ] Run keypoint quality and refinement without changing source crop lineage.
- [ ] Only after keypoint parity passes, exercise subject-mask consumers through
  the same crop/provider contract.
- [ ] Keep training exports dense/materialized and record the provider identity
  used to materialize each training image.

## Phase 5: Registry and campaign orchestration

- [ ] Add separate registry/readiness state for:

```text
acquisition_crop_inventory
acquisition_crop_canonical_import
acquisition_live_detection_import
crop_sufficiency_comparison
canonical_crop_rowset
crop_pixel_routing
keypoint_provider_consumption
keypoint_completion
subject_mask_provider_consumption
```

- [ ] Do not collapse these into `crop_stream_available` or `crop=ok`.
- [ ] Keep stream availability separate from model consumption.
- [ ] Make the registry project immutable Zarr state; do not make SQLite the
  scientific authority.
- [ ] Have parallel jobs write per-recording immutable JSON result receipts.
- [ ] Reconcile campaign receipts into a JSONL ledger after jobs finish.
- [ ] Run one controlled registry refresh/publication step from successful
  receipts instead of allowing array workers to contend on SQLite.
- [ ] Preserve failed and skipped receipts with explicit reason codes.
- [ ] Ensure monitor terminal state is based on terminal receipts rather than an
  output-exists/status-update race.
- [ ] Use one writer per analysis Zarr at a time. Jobs for different recordings
  may run concurrently after measuring PRFS pressure.
- [ ] Separate metadata/CSV import concurrency from crop-video decode and media
  hashing concurrency; they have different I/O costs.

## Phase 6: Focused tests

### Import and identity

- [ ] Full and crop stream declarations bind the same recording, camera, frame
  clock, and native full-frame extent.
- [ ] Complete frame ledgers preserve blank and no-detection rows.
- [ ] Crop-video row order, recording-frame identity, and Orange local IDs remain
  distinct.
- [ ] `crop_xywh` and selected live `detection_xywh` remain distinct.
- [ ] Missing/corrupt manifest, CSV, summary, status, keyframe, checksum, or
  media identity fails closed.
- [ ] Wrong camera, recording, extent, coordinate space, or frame cardinality
  fails closed.
- [ ] Exact repeat import is a no-op; differing reimport cannot overwrite an
  immutable generation.
- [ ] Consolidated metadata exposes the completed import only after final
  publication.

### Routing

- [ ] Asymmetric off-center fixtures receive no flip or reflection.
- [ ] Boundary-inclusive refined-bbox/context containment in the recorded
  acquisition crop is specified and tested.
- [ ] Canonical crop rows retain exact refined-detection instance keys.
- [ ] Complete acquisition crop frames map exactly back to native full-frame
  coordinates through their recorded 384 x 384 origin.
- [ ] Blank/no-detection/missing/insufficient crop rows select the correct
  full-frame fallback reason.
- [ ] Duplicate, missing, stale, partial, or reordered raw-stream/provider rows
  fail closed.
- [ ] Empty refined-detection recordings produce a valid zero-row crop/provider
  publication.
- [ ] Raw acquisition arrays and existing immutable runs remain unchanged after
  successor publication.

### Consumers

- [ ] Direct acquisition decode and equivalent full-frame crop use the same
  canonical geometry and output dimensions.
- [ ] Mixed provider batches preserve requested crop-run row order.
- [ ] Keypoint outputs carry exact crop row and instance identity.
- [ ] Full-frame keypoint projection is correct for acquisition and fallback
  rows.
- [ ] Keypoint finalization rejects stale or incomplete provider evidence.
- [ ] Subject-mask consumers cannot bypass the selected crop/provider manifest.
- [ ] Crimson can read crop geometry and derived outputs without parsing Orange
  CSVs.
- [ ] Crimson raw-stream playback, if implemented, follows the Zarr stream
  contract and exact media identity.

## Phase 7: Canary sequence

Use commit-pinned, selector-ineligible outputs until all required checks pass.

1. [ ] Rerun readiness with a 384-pixel minimum and run schema/import dry-run on
   the representative arena-1 canary.
2. [ ] Publish a new immutable raw crop-stream generation in that canonical
   analysis Zarr.
3. [ ] Validate direct and consolidated metadata views.
4. [ ] Compare every imported Zarr ledger value with the producer CSV.
5. [ ] Build the canonical refined-detection crop rowset and provider routing.
6. [ ] Measure acquisition-selected versus full-frame-fallback counts and inspect
   temporal clusters of fallback.
7. [ ] Decode a deterministic sample from each provider and verify placement.
8. [ ] Run a bounded keypoint inference canary.
9. [ ] Compare keypoint output with the existing flat/full-frame path using a
   frozen row sample and evidence-derived tolerances.
10. [ ] Complete Crimson exact-reader and visualization checks.
11. [ ] Run focused local tests, the optimized required CI shards, and any
    required cluster canary.
12. [ ] Only after CI is green, publish a production-eligible selector in a new
    commit-pinned deployment.

## Phase 8: GoodBatBadBat rollout

- [ ] Regenerate the read-only 84-recording inventory immediately before
  planning.
- [ ] Freeze exact analysis Zarr, stream, refined-detection, geometry, and media
  identities in the campaign plan.
- [ ] Exclude or explicitly review any recording whose state changed since the
  inventory snapshot.
- [ ] Publish complete raw acquisition crop-stream imports across the corpus.
- [ ] Validate and reconcile import receipts before launching crop routing.
- [ ] Publish canonical crop/provider successors across the corpus.
- [ ] Summarize per-recording acquisition-use/fallback/unrecoverable counts.
- [ ] Stop records with unresolved provider rows; do not silently drop them.
- [ ] Launch keypoint work only for recordings with complete validated routing.
- [ ] Refresh the registry once per completed campaign tranche from durable
  receipts.
- [ ] Produce a final corpus report with exact run IDs/digests, row counts,
  provider fractions, failures, timings, and registry status.
- [ ] Confirm source videos, producer metadata, raw detections, reviewed refined
  detections, geometry artifacts, and existing immutable runs were not rewritten.

## Acceptance criteria

The implementation is production-ready for GoodBatBadBat only when:

- every in-scope recording has a validated complete raw acquisition crop-stream
  Zarr import, including blank/no-detection rows;
- raw stream identity is explicit and sidecars are content-bound;
- every selected refined-detection instance maps to exactly one canonical crop
  row and one valid pixel provider;
- every new GoodBatBadBat crop row declares `zebrafish_crop_384_v1` and has
  384 x 384 persisted geometry;
- complete acquisition crop frames are used only when exact frame,
  dimensions, refined-bbox, and context-containment policy passes;
- every other recoverable row is supplied from the retained full-frame video;
- provider routing and model outputs preserve ordered instance/crop-row identity;
- downstream runs record exactly which pixel source each row used;
- Crimson can consume the stable Zarr contracts without Orange-specific CSV
  parsing;
- registry state distinguishes availability, canonicalization, routing, and
  consumption;
- required CI and commit-pinned canary evidence are green;
- no current GoodBatBadBat source or immutable artifact was modified or deleted.

## Deferred crop-only work

Current GoodBatBadBat remains `full_plus_crop`. Future crop-only certification,
sparse full-frame audit imagery, retention decisions, and destructive tooling
are governed by [Crop-only recording storage profile](crop_only_recording_storage_profile.md)
and are not part of this production campaign.

## Related contracts

- [Acquisition video stream source policy](acquisition_video_stream_source_policy.md)
- [Acquisition crop-video ROI provider plan](acquisition_crop_video_roi_provider_plan.md)
- [Crop geometry storage contract v1](crop_geometry_storage_contract_v1.md)
- [Crop pixel work-package contract](crop_pixel_work_package_contract.md)
- [Frame domains resolver design](frame_domains_resolver_design.md)
- [Detection analysis run surfaces](detection_analysis_run_surfaces.md)
- [Recording-bound geometry import and validation design](recording_bound_geometry_import_and_validation_design.md)
