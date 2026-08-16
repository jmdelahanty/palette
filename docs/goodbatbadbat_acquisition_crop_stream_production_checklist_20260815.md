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

### Selector-ineligible hybrid-provider canary evidence

Commit-pinned real-data validation completed on 2026-08-15 at commit
`a8aa4555b9f4cd48f715355872c6eb97760a8e78`. The canonical representative
analysis Zarr and both source MP4s were read-only. All writes were confined to:

```text
/groups/johnson/johnsonlab/jeremy/staging/
palette_hybrid_crop_provider_canary_20260815_40b597e6/
```

The disposable overlay bound:

- canonical acquisition-ledger record SHA-256
  `9c8d72952633e6991e8d6f2bb34f3b22ec4b2437be03b2d81f7acc1fdd4039d8`;
- strict finalized refined run
  `refined_detect_goodbatbadbat_geometry_production_400fce8f`;
- refined manifest digest
  `18b65cf76dc90bce32dafc5369be3741d38ac72783ad5d8f9d021ba96c5143cd`;
- refined logical-content digest
  `36ef5c8c48c7635138cf6707d7c6150677a6a5f5c759caf6c997aba60db1429e`.

LSF job `153425527` completed successfully in 54.99 seconds on an L4 host. It
published selector-ineligible run
`crop_hybrid_goodbatbadbat_cluster_canary_v2` with provider-record SHA-256
`30f8d3b3f5564a6578e3d0c36f27eba1c375bc7205ca8ed2b1e69801948df9ca`.
Independent direct and consolidated readers both validated that digest and the
complete 151,478-row provider record. Routing was:

| Route | Rows | Fraction |
| --- | ---: | ---: |
| Acquisition crop video | 149,440 | 98.654590% |
| Full-frame supplemental cache | 2,038 | 1.345410% |
| Missing/unrecoverable | 0 | 0% |

Every fallback reason was `blank_acquisition_crop`. The supplemental payload
contained only those 2,038 rows and occupied 300,515,328 bytes; the 149,440
acquisition-backed images were not duplicated into the Zarr or cache.

The first real writer pass exposed that the completed run was absent from the
archive's older consolidated generation. Commit `a8aa4555` makes root
consolidation and consolidated provider-digest visibility mandatory final
publication steps, and marks/reconsolidates the run as failed if sealing fails.
The second writer pass above proved the fixed behavior.

LSF job `153425528` then decoded crop rows 3,982 through 3,984, whose source
codes are acquisition, supplemental, acquisition, as one ordered
`uint8[3,384,384]` batch through `CropImageSource`. The output used pixel
contract `orange_mono_pynvvc_luma_hybrid_uint8_v1`, had a nonempty value range
of 31 through 231, and completed with return code zero. Required CI remains
unrun for `a8aa4555`; this evidence does not make the branch merge-ready or
authorize production selectors.

### Signed 384 x 384 keypoint canary and model-input policy

A successor selector-ineligible canary completed on 2026-08-15 at commit
`8380406ed29c2d160c2831abfe7c77abc9dadab9`. It used a signed hybrid-provider
rowset and a balanced 256-row work package containing 128 acquisition-video
rows and 128 full-frame fallback rows. The exact keypoint model was
`pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2`, with model
SHA-256
`cce63d534a8f1491db1e2c71cb9236768c445722013dc39faeaf62a9d0a9a377`.

The three relevant image extents were all 384 x 384:

| Extent | Canary value | Meaning |
| --- | --- | --- |
| Persisted crop geometry | 384 x 384 | The canonical zebrafish ROI and coordinate frame |
| Submitted tensor | 384 x 384 | The array passed to the pose runtime |
| Network input | 384 x 384 | The stride-aligned extent consumed by the model |

Because 384 is divisible by the model's verified stride of 32, this profile
uses an identity spatial transform and requires no model-input padding. This is
separate from the crop builder's zero fill at source-image boundaries: crop
boundary fill preserves the canonical 384 x 384 ROI, whereas model-input
padding exists only to satisfy a declared network-input extent.

The model's historical training images were stored at 512 x 512, but the
Ultralytics training pipeline submitted resized 256 x 256 tensors to the
network. A convolutional pose model can execute at other stride-compatible
spatial extents; architectural compatibility does not establish scientific
accuracy at that extent. The bounded 384 x 384 canary therefore supplies
runtime and preliminary empirical evidence, not production approval.

The terminal canary produced 253 successful poses from 256 rows (98.83%):
125 of 128 acquisition-backed rows and 128 of 128 fallback rows. The three
failures were explicit `no_pose_detection_above_threshold` outcomes. Ordered
crop rows, instance keys, source-row signatures, provider digest, and the exact
work-package pixel digest were preserved. The run remained under
`keypoint_shard_runs`, declared `stage_selector_eligible=false`, and did not
modify a canonical recording or selector.

Before production use:

- publish the bounded pixels, row identities, keypoint arrays, result receipt,
  and review montage in a durable benchmark location outside staging;
- visually inspect deterministic acquisition and fallback samples, all three
  failures, and low-confidence successes;
- record an explicit accept/reject decision with the reviewed artifact digests;
- publish a successor to `pose_model_input_contract_v2.json` rather than
  rewriting that immutable evidence, adding an exact 384 x 384 identity profile
  bound to Ultralytics 8.3.169 and the reviewed canary;
- make target manifests and production recipes require that exact successor
  contract and profile; and
- complete required CI before any selector, shared-checkout, or production
  campaign activation.

For future model training, retain both 512 x 512 and 384 x 384 native examples
with their original source geometry and reversible preprocessing provenance.
Use 384 x 384 as the primary zebrafish deployment profile. Mixed-size training
may vary stride-aligned input size between batches, but every tensor in one
batch must share a shape. Source balance and per-profile validation must be
reported separately so a large legacy 512 corpus cannot hide regressions on
the 384 acquisition profile. A future model is multi-size-capable only after
each intended deployment profile has its own accepted runtime and scientific
validation evidence.

The bounded evidence was durably published after visual inspection at:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/
goodbatbadbat_hybrid_pose_384_canary_20260815_v1/
```

The review bundle was produced by clean commit
`8d053e4cd499cb92855ca8c797e9c64e2e079904`. Its review-record digest is
`1652839606af97ce2c145451b8979441270c0f17e2050d3cc3de158be1f9a13e`;
the copied 37,748,736-byte pixel payload retained SHA-256
`8327b37c341b12379ae379ada2fc478a66267ddf76a864209d80007ee014fccc`.
The bundle contains all 256 pixels and identities, extracted keypoint arrays,
the original canary receipt, an all-row overview, a failure/low-confidence
montage, and a provider-spread montage.

The agent visual screen decision digest is
`3da2964eba3c8601342587065625de9af89549dd74fba40481a10133c083ab09`.
All five landmarks were anatomically consistent across reviewed acquisition and
fallback successes. No crop-origin shift, scale error, provider-specific
coordinate offset, axis flip, or systematic snout-tail reversal was visible.
The three failures were explicit misses rather than fabricated coordinates;
two overlapped the dish rim and one otherwise visible fish was missed.

That decision is deliberately
`pass_for_successor_384_profile_and_full_recording_canary`. It does not authorize
a production selector or corpus campaign. It required a successor immutable
model-input contract followed by one complete selector-ineligible recording
through terminal inference, strict candidate finalization, keypoint quality,
and refinement. Operator confirmation and required CI remain mandatory before
the 84-recording campaign.

The immutable successor contract is
`pose_model_input_contract_v2_goodbatbadbat_384_v1.json`, with file SHA-256
`f95265c0708063af7ac9d03dc68435a85ff17fa619be102fb99de13d4988b928`
and payload digest
`cd2b6050ef24cbcaf70cb5c73a4812225077739c5639ba202d701d6e4ca568ef`.
It approves only Ultralytics 8.3.169 for the 384 x 384 profile because that is
the exact runtime used by the reviewed LSF canary. The workstation environment
currently resolves 8.3.214 and therefore fails this profile's runtime check by
design. A cheap LSF runtime-and-contract preflight must precede the complete
recording canary; do not add 8.3.214 without separate empirical evidence.

### Complete selector-ineligible 384 x 384 recording canary

The complete arena-1 canary passed on 2026-08-15/16 using L4 inference job
`153427334` and clean commit
`32307bffe161082377baa6e0704659ec05d9e101`. The job processed all 151,478
ordered provider rows: 150,180 poses succeeded and 1,298 remained explicit
terminal inference misses. Inference throughput was approximately 476 rows per
second. The sealed terminal receipt digest is
`46cfa8f2de55b95db96c165806076aeb0daac9d046bc812039ede6556224e08a`.

The canary exposed and closed one contract gap before final publication. A
generic crop-v2 candidate recenters every 384 x 384 window on the offline
refined detection, but acquisition-backed rows intentionally use Orange's
recorded 384 x 384 window and full-frame origin. Strict crop policy v2 now
supports `verified_explicit_per_row` placement. It binds the signed hybrid
provider and requires exact ordered instance, refined-row, frame,
acquisition-frame, origin, and size equality. Existing center-derived policy-v1
payloads remain unchanged.

The validated strict crop candidate is:

```text
crop_runs/crop_goodbatbadbat_geometry_384_hybrid_32307bff
manifest digest: f2be8ce38d4610eefd06c4144f48401ab78e17e2243d812efe532485b84be4b9
```

It binds hybrid provider record
`02dd09050ddada64239f12f846bde2c15b2004954d207c7397175cb0023f6d8f`,
rowset fingerprint
`414a9fd98bfc87d5cbedf6aee07e34743d6de1b3518f5f3c80abbd472b6d5c9d`,
and pixel fingerprint
`88006b24ee596ea5ac695dff8626796351d7b66b958f8a037f555bfcc8427d49`.

The terminal result then finalized into these four immutable candidates:

```text
keypoints_runs/keypoints_goodbatbadbat_hybrid_pose_384_full_canary_20260815_v4
keypoint_quality_runs/keypoint_quality_goodbatbadbat_hybrid_pose_384_full_canary_20260815_v4
refined_keypoints_runs/refined_keypoints_goodbatbadbat_hybrid_pose_384_full_canary_20260815_v4
analysis/body_frame_runs/body_frame_goodbatbadbat_hybrid_pose_384_full_canary_20260815_v4
```

Independent direct and consolidated reads found 151,478 rows in every stage,
identical manifests, exact instance-key coverage against both strict crop-v2
and the signed provider, 150,180 quality/refinement/body-frame successes, no
manifest errors, and no selector changes. The reconstructable 22,336,339,968-
byte NRS flat cache and its 9,774-byte manifest were deleted only after that
audit; the sealed terminal artifact and all candidates remain.

All writes remain confined to the disposable overlay and durable operations
receipts. The canonical recording, registry, production selectors, source
media, and shared `/groups` checkout were not changed. Required CI remains
unrun, so this branch and its candidates are not merge-ready or
production-eligible.

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
- The hybrid provider builder now defaults to the pointer-selected canonical
  ledger and joins it to strict finalized refined detections by recording frame.
  Its historical instance-key join is available only through the explicit
  `legacy_crop_run` compatibility mode.
- Hybrid crop runs carry a digest-bound provider record over exact refined-row,
  ledger-row, source-kind, reason-code, and ROI arrays. Whole-recording keypoint
  planning and terminal startup validate that exact digest when a hybrid run is
  configured.
- Hybrid crop publication now consolidates the archive only after all payload,
  attrs, provenance, completion, and optional nonauthoritative `latest_any`
  writes, then verifies that the consolidated generation exposes the exact
  provider digest.
- Registry schema 69 projects `crop_pixel_routing_ready`, provider identity,
  crop/routing policies, and acquisition-versus-recovery counts separately from
  raw crop-stream availability.

### Missing or not production-complete

- The implementation branch has a commit-pinned experimental deployment but is
  not production-deployed and has not passed required CI.
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
- The old acquisition crop-run geometry remains based on the selected live box
  and is compatibility-only. New hybrid provider rows are bound to the selected
  offline refined-detection rowset.
- Acquisition detection import remains a nonselector
  `detection_artifact_runs` compatibility surface; canonical promotion is not
  implemented.
- Whole-recording keypoint orchestration now pins the exact hybrid provider
  record, but still materializes/uses a whole-rowset flat ROI cache for terminal
  inference. Direct grouped provider-block execution remains future work.
- A signed balanced 384 x 384 work-package canary and durable visual screen
  passed. Its source terminal remains selector-ineligible staging evidence; a
  successor model-input contract and complete recording canary remain pending.
- Registry readiness now expresses raw-stream canonicalization and routing
  completeness; completed keypoint provider consumption remains future work.
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
- [x] Define a versioned crop-pixel routing policy and reason-code vocabulary.
- [x] Define `zebrafish_crop_384_v1` as a species-aware 384 x 384 crop geometry
  profile rather than changing an unexplained global integer default.
- [x] Inventory and update all production-facing 348 defaults, including shared
  crop defaults, default configuration, geometry-review approval planning,
  readiness reporting, preflight, and crop publication entry points.
- [x] Keep persisted crop geometry separate from model tensor size. Reuse the
  existing reversible `ModelInputTransform` for identity or centered zero-padding
  to a larger submitted extent, and retain its declared preprocessing identity.
- [x] Preserve existing 348 x 348 runs unchanged and classify them through their
  original crop-policy identity.
- [x] Freeze full-frame edge handling for 384 x 384 fallback crops, including
  whether the ROI origin is clamped or pixels outside the native extent are
  zero-filled. `zebrafish_crop_384_v1` uses translation-only centering and
  zero-fills pixels outside the native extent without changing coordinates.
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

- [x] Resolve one exact, complete, finalized immutable refined-detection run and
  its exact dish-gate/selection lineage.
- [x] Build one canonical 384 x 384 crop row per selected refined-detection
  instance using `zebrafish_crop_384_v1`.
- [x] Preserve the refined-detection `instance_key` as the crop observation key;
  do not mint a replacement identity from the live acquisition box.
- [x] Require identical ordered `instance_key` coverage between the selected
  refined instances and the derived crop rowset.
- [x] Preserve `source_refined_row_ids`, frame indices, acquisition-frame
  indices, full-frame ROI placement, bbox geometry, and crop-policy identity.
- [x] Extend strict crop policy identity without changing existing v1 bytes:
  refined-centered runs retain policy v1, while acquisition/hybrid runs use
  policy v2 with `verified_explicit_per_row` origins bound to the exact signed
  provider record, rowset fingerprint, pixel fingerprint, and row-signature
  specification digest.
- [x] Require the explicit-origin provider to have identical ordered instance,
  refined-row, frame, acquisition-frame, and 384 x 384 size coverage before its
  recorded origins can enter crop-v2. Revalidate that authority before and
  after atomic publication.

### Acquisition-video eligibility

For each canonical crop row:

- [x] Resolve exactly one raw acquisition crop ledger row for the same recording
  frame.
- [ ] Require a decodable, nonblank crop-video frame with exact 384 x 384
  dimensions and valid full-frame placement.
- [x] Treat the complete recorded 384 x 384 acquisition crop window as the
  candidate ROI; do not take a smaller subwindow.
- [x] Test whether the selected refined fish bbox and the versioned required
  context margin are contained within that recorded crop window.
- [x] When eligible, use the complete crop-video frame and record its exact
  native full-frame origin from `source_crop_xywh`.
- [x] Do not use the selected live bbox as the canonical analysis bbox.
- [x] Record full-precision containment margins and policy result.

### Full-frame fallback

- [x] Route blank, missing, ambiguous, or insufficient acquisition
  crops to a 384 x 384 full-frame recovery ROI generated from the canonical
  refined detection.
- [x] Materialize only fallback rows into a supplemental flat cache. Production
  cluster execution must place its work/cache path according to the job plan;
  the disposable canary uses `/tmp`.
- [ ] Bind the cache manifest to the exact full-frame media identity, crop run,
  row IDs, crop policy, and software identity.
- [x] Preserve explicit routing reasons for every statically classifiable route,
  including:

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

- [x] Fail closed if any canonical row has no valid provider. Decode failures
  currently fail the publication rather than silently changing a frozen route;
  `acquisition_crop_decode_failed` remains reserved for a later prevalidated
  decode-evidence generation.

### Immutable routing artifact

- [x] Publish the provider decision as part of the canonical crop run or as one
  exactly bound immutable provider manifest.
- [x] Store source kind, raw stream row, crop-video frame, full-frame ROI origin,
  supplemental-cache row, and reason code per crop row.
- [x] Bind the exact raw acquisition stream generation, refined detection run,
  full-frame source, crop policy, and geometry selection digests.
- [x] Ensure provider policy can change by publishing a successor without
  rewriting raw acquisition arrays or refined detections.

## Phase 4: Production model-consumer integration

- [ ] Generalize the whole-recording keypoint planner from mandatory flat-cache
  input to an explicit ROI provider manifest.
- [ ] Inventory the exact pose-model input contract for every selected zebrafish
  keypoint/mask model.
- [x] Feed native 384 x 384 input directly when model and stride contracts permit
  it; otherwise use the existing centered `pad_to_size` transform for a larger
  submitted extent and its existing inverse coordinate mapping.
- [x] Prove selector-ineligible 384 x 384 tensor/runtime execution on a balanced
  acquisition-plus-fallback work package with an exact signed provider rowset.
- [x] Publish the bounded canary evidence and montage durably, complete visual
  review, and freeze the decision in a checksummed review receipt.
- [x] Add an exact accepted 384 x 384 identity profile in a successor immutable
  model-input contract; do not broaden or rewrite the existing 352 profile.
- [ ] Record any later Ultralytics/network resize as the separate framework
  preprocessing stage already required by the pose model-input contract.
- [ ] Do not silently center-crop acquisition frames back to 348 x 348.
- [ ] Support provider blocks for:
  - complete 384 x 384 acquisition crop-video frames;
  - full-frame supplemental flat-cache rows.
- [ ] Group work by provider and decode source rather than switching decoders
  row by row.
- [x] Merge model outputs back into canonical crop-run row order.
- [x] Require exact `source_crop_row_ids` and `instance_key` coverage in raw
  keypoint output.
- [x] Extend terminal receipts to bind crop run, provider
  manifest, source media, rowset digest, model, and configuration.
- [x] Make stale, partial, duplicated, reordered, or mismatched provider rows
  block publication.
- [ ] Record whether each output row used acquisition crop pixels or full-frame
  recovery.
- [x] Preserve full-frame keypoint coordinates by projecting through canonical
  crop placement, not through the live detection box.
- [x] Run keypoint quality and refinement without changing source crop lineage.
- [x] Make whole-recording subject-mask targets distinguish the pixel-source
  `crop_run` from an explicit strict `geometry_crop_run`. A single run may fill
  both roles only when it is itself the coordinate-aware crop-v2 authority.
- [x] Bind full-recording raw mask inference and later publication to the same
  digest-bound expected-work-unit manifest, including exact frame and crop-row
  coverage. Older raw drafts without that receipt remain unpublishable.
- [ ] Only after keypoint parity passes, exercise subject-mask consumers through
  the same crop/provider contract.
- [ ] Keep training exports dense/materialized and record the provider identity
  used to materialize each training image.

## Phase 5: Registry and campaign orchestration

- [x] Add separate registry/readiness state for raw acquisition crop import and
  crop-pixel routing. Keypoint/subject-mask consumption states remain pending:

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
- [x] Strict crop-v2 publication preserves signed acquisition/hybrid per-row
  origins instead of silently recentering them on the offline detection.
- [x] Existing center-derived crop-v2 manifests round-trip byte-for-byte through
  the unchanged policy-v1 parser and writer.
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
5. [x] Build the canonical refined-detection crop rowset and provider routing.
6. [x] Measure acquisition-selected versus full-frame-fallback counts and inspect
   temporal clusters of fallback.
7. [x] Decode a deterministic sample from each provider and verify placement.
8. [x] Run a bounded, signed, balanced keypoint inference canary at 384 x 384.
9. [x] Publish and visually review the bounded keypoint evidence, including all
   failures and deterministic samples from both pixel providers.
10. [x] Run one complete selector-ineligible recording through strict crop-v2,
    terminal inference, keypoint quality, refinement, and body-frame publication.
11. [ ] Run one full-recording subject-mask inference, refinement, quality, and
    inactive bundle-publication canary with separate pixel and geometry crop
    authorities plus exact work-unit coverage.
12. [ ] Compare keypoint output with the existing flat/full-frame path using a
   frozen row sample and evidence-derived tolerances.
13. [ ] Complete Crimson exact-reader and visualization checks.
14. [ ] Run focused local tests, the optimized required CI shards, and any
    required cluster canary.
15. [ ] Only after CI is green, publish a production-eligible selector in a new
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
