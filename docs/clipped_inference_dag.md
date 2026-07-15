# Registry-pinned clipped inference DAG

`fisheye.cluster.clipped_inference` composes one immutable LSF workflow for a
clipped recording cohort. It covers detection, detection refinement, finalized
collection creation, flat ROI cache construction, proxy crop binding,
keypoints, keypoint refinement, subject-mask inference, refined subject masks,
content validation, registry reconciliation, and optional NRS cleanup.

The target manifest schema is `palette.clipped_inference_targets.v1`. Every
target must pin its registry `recording_id`, recording directory, and canonical
analysis Zarr. The command also requires exact registry set and run identifiers
for detection, pose, and subject-mask models. Planning verifies the registered
paths and model SHA-256 values and refuses output collisions.

If a campaign stops after its detection artifacts were atomically imported,
`--resume-existing-detections` permits a new immutable run root to reuse those
groups without rerunning YOLO. The run label must remain unchanged so the
planned detection identities remain unchanged. Planning validates every import
receipt and run-group tree (excluding the ephemeral source tarball), requires a
complete run, and compares the embedded model, workflow, recording, clip,
camera, video, target, and run-name provenance to the new plan. Every later
output must still be absent. The DAG then submits short CPU revalidation jobs
in place of GPU inference jobs and continues through detection refinement.

## Execution contract

All production work is represented as structured `LsfJob` objects and wrapped
by the durable LSF runtime-status envelope. `--dry-run` writes `plan.json`,
`lsf_plan.json`, and per-target detection plans without calling `bsub`.
`--apply` performs the same planning on the workstation and sends only the
individual `bsub` commands through `login1-citrus-poller` (or
`--submit-host`). No inference, finalization, validation, or Zarr mutation runs
on the workstation or login poller.

For each recording, the dependency structure is:

```text
22 detect -> sharded recording-order quality source
          -> collection quality reconcile -> 22 keyed detect refine
                                          -> finalized detection collection
                                      |
                               6 cache bundles
                                      |
                              proxy crop binding
                                /             \
                       22 keypoints       22 subject masks
                              |                  |
                    keypoint finalize           |
                              |                  |
                     keypoint refine            |
                               \                /
                           22 mask packages
                                  |
                         refined-mask import
                                  |
                       exact content validation
```

The cache is the physical crop materialization. The proxy crop runs bind its
rows to the canonical Zarr; there is no redundant standalone crop-image write.
Keypoints and masks safely read the same immutable cache concurrently. Each
mask package waits for both its clip-local probability shard and the exact
recording-level refined-keypoint run used for left/right eye assignment.

Detection quality is a recording-level stage, not 22 independent clip-local
state machines. The source materializer streams clip-local raw boxes, stable
`instance_key` values, and canonical parent-frame indices into one immutable
indexed-sharded `detect_collection_sources` run. Parallel quality workers emit
compact traces; one reconciler carries temporal state across every shard and
clip boundary and publishes `detect_quality_runs/<run>`. Each clip refine job
reads only its declared source slice and requires exact key equality before
using those labels. Historical nested `quality_reports` remain a read fallback
for old single-run archives.

The manifest always contains a nonempty `targets` list. A one-recording run is
therefore a one-item campaign, while a multi-recording run expands this same
subgraph for every item. `--max-active-targets` bounds recording concurrency by
gating a later target's detection jobs on an earlier target's successful
validation; it does not select a different workflow implementation. Each
target may declare `expected_subject_count` (default `1`) for its quality
policy.

Every proxy crop run also carries the full-resolution source-video dimension
contract: `source_video_width`, `source_video_height`, `width`, and `height`.
Proxy creation resolves one consistent value from the analysis root, the
refined detection run, or its explicitly bound raw detection run. It refuses
creation if no dimensions can be established or if clips disagree. This is
required even though the pixels come from the ROI cache: keypoint coordinates
must still be transformed into the full-frame coordinate system. The merged
collection proxy preserves the same dimensions.

Detection and keypoint model outputs use the repository's indexed-sharding
defaults. Raw subject-mask probabilities are `uint8` encoded probabilities
with 32-row inner chunks and 2,048-row storage shards. Refined subject masks
are imported as authoritative dense binary `uint8 masks_roi`, with sampled
contours and eye geometry enabled and full ragged contours disabled.

The imported-run validator reports top-level success as `status=ok`; nested
checks report `status=pass`. Detection work-unit orchestration accepts both
success vocabularies so it does not reject an otherwise valid atomic import.

Validation requires all planned collection/cache/run identities, modern unique
`instance_key` arrays, equal row counts across refined detections, caches,
keypoints, and refined masks, non-binary/nonempty encoded raw probabilities,
binary/nonempty dense refined masks, sampled contours, and exact refined
keypoint assignment lineage. A single registry writer reconciles all targets
only after every validation succeeds. NRS cache and package directories are
removed only after that registry report passes its integrity check.

## Keypoint-stage recovery

`fisheye.cluster.clipped_inference_keypoint_recovery` resumes a failed campaign
after ROI caches and raw subject-mask probability shards are complete. It does
not rerun detection, detection refinement, cache construction, or subject-mask
inference. Planning first runs the read-only
`fisheye.utils.prepare_clipped_keypoint_recovery` preflight. For every planned
clip, the preflight requires:

- a complete cache manifest and alias with a valid row count;
- an exact, complete raw subject-mask shard with matching crop, collection,
  cache, model path, and model SHA-256 provenance;
- equal cache and raw-probability row counts;
- a repairable per-clip proxy source-dimension contract;
- no complete keypoint shard or downstream output collision.

The submitted recovery starts with one short CPU maintenance job. That job
repairs the proxy attributes, removes only the exact incomplete keypoint shard
groups named by the source plan, and clears `latest_pending` only if it selects
one of those incomplete groups. The remaining DAG is:

```text
repair/cleanup -> 22 keypoints -> keypoint finalize -> keypoint refine
                                                        |
                                  completed raw masks -> 22 mask packages
                                                        |
                                          refined-mask import
                                                        |
                                      validation -> registry -> NRS cleanup
```

The merged proxy may legitimately be absent during preflight because the
keypoint finalizer creates it immediately before merging the completed shards.

Collection-level refined-mask import runs on the `local` CPU queue with a
three-hour requested wall time. The import publishes a very large dense mask
surface and can exceed the `short` queue's hard one-hour limit even when memory
use is low. If import is interrupted after all clip packages complete,
`fisheye.cluster.clipped_inference_import_recovery` preflights those packages
and complete refined keypoints, then submits only import, validation, registry
reconciliation, and NRS cleanup. Recovery imports use `--overwrite` to replace
the incomplete, non-promoted collection output; a complete output is refused.

## Next-recording encoded-chunk publication checklist

The current v1 clip packages restart their Zarr row grid at local row zero.
When a package row count is not a multiple of the canonical dense-mask row
chunk, all following encoded package chunks are shifted relative to the
recording-level grid. The compatibility importer must then decode, splice, and
re-encode the dense masks. The following checklist gates a v2 package contract
that can publish encoded chunks directly while retaining an ordinary,
editable, non-sharded canonical `masks_roi` array.

### Canonical grid and ownership

- [x] Freeze the recording-level ordered `source_crop_row_ids`, total row
  count, label order, dtype, fill value, chunk shape, and codec configuration
  before clip finalization starts.
- [x] Fingerprint that complete array contract and require the same fingerprint
  in every package manifest.
- [x] Assign exactly one writer to every physical canonical chunk
  `(row_chunk, channel_chunk, y_chunk, x_chunk)`; never allow two jobs to write
  different logical rows within one physical chunk.
- [x] Store interior package chunks under their global canonical chunk keys
  instead of restarting the chunk grid at package-local row zero.
- [x] Define the cross-clip boundary policy explicitly. Prefer complete
  globally aligned interior chunks plus a small boundary-fragment merge job;
  at most the chunks intersecting clip boundaries should require pixel decode.
- [x] Record requested and effective worker/chunk ownership in provenance.

### Package v2 contract

- [x] Add a versioned manifest containing the global array-contract
  fingerprint, global row interval, owned canonical chunk keys, boundary
  fragments, per-object size/digest, source runs, and package completion
  status.
- [x] Require complete refined-keypoint assignment lineage and raw-mask source
  lineage before a package may be sealed.
- [x] Keep authoritative dense binary `uint8 masks_roi` payloads. Bitpacked,
  RLE, metrics, geometry, and sampled contours remain derived data and must not
  replace the dense edit surface.
- [x] Preserve the v1 package reader/importer as a compatibility fallback.

### Transactional publisher

- [x] Preflight exact chunk-key coverage, uniqueness, contract fingerprints,
  object digests, and boundary-fragment coverage before mutating the canonical
  recording Zarr.
- [x] Publish into a new incomplete run group, copy encoded objects without
  Zarr array assignment, write canonical metadata, then validate before
  setting completion and latest/review pointers.
- [x] Refuse overwrite of complete runs and remove an incomplete destination
  only through the explicit recovery path.
- [ ] Make retries idempotent and record whether each object was copied,
  verified existing, or boundary-merged.
- [x] Run registry reconciliation and NRS cleanup only after content
  validation succeeds.

### Canary and rollout gates

- [x] Build a two-clip canary whose boundary is deliberately not divisible by
  the 128-row dense-mask chunk size.
- [ ] A/B the v1 decoded importer and v2 encoded publisher for exact row
  identity, decoded mask equality, metrics, sampled contours, eye geometry,
  provenance, completion pointers, and Crimson/`MaskStore` reads.
- [ ] Validate every boundary row plus deterministic samples from every
  package and channel; verify copied encoded-object digests before decode
  checks.
- [ ] Record wall time, CPU time, peak RSS, bytes read/written, and PRFS object
  operations for both paths.
- [x] Keep v2 behind an explicit feature flag for the first full recording and
  fall back closed to the v1 importer on any contract mismatch.
- [ ] Enable v2 by default only after a full recording passes validation,
  registry reconciliation, Crimson loading, and post-cleanup audit.

Independent of package v2, the v1 importer must pre-index package row
placements once per destination chunk grid. It must not rescan all recording
rows for every destination chunk.

As of 2026-07-14, the checked implementation items above are covered by the
v2 grid/package/publisher code and a deliberately misaligned in-memory two-clip
A/B test. The production flag `--encoded-mask-packages` remains off by default.
The unchecked retry-accounting, cluster performance, full scientific parity,
Crimson, and default-rollout gates require the PRFS two-clip canary and then one
complete recording.

## Sleepyfish dry run, 2026-07-14

The reviewed three-recording plan covers cams 2010093, 2010094, and 2010096,
22 clips each. It pins:

- Detection set `detect_all_available_detect_training_v003`, run
  `detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1`.
  This keeps the new cohort consistent with completed cam2010095 and avoids
  camera-dependent automatic ranking.
- Pose set `pose_all_registry_reviewed_v2_keypoints_20260520_v001`, run
  `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2`.
- Subject-mask set
  `subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001`,
  run `subject_masks_union_all_components_v001`.

The dry run rendered 368 LSF jobs: 122 jobs per target, one all-target registry
job, and one NRS cleanup job. The audit confirmed 66 clips, exact two-way mask
package joins, shared proxy gates for the parallel keypoint/mask branches,
direct cache execution with no nested `bsub`, runtime wrapping on every job,
and no submission evidence. The reviewed snapshot is under
`/tmp/sleepyfish_clipped_full_20260714_v001_dryrun`; it is diagnostic evidence,
not a durable campaign directory.
