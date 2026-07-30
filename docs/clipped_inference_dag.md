# Registry-pinned clipped inference DAG

> **Architecture boundary:** this is the clipped-recording recipe built on the
> general structured LSF production engine. It is not intended to make the
> engine exclusive to clipped recordings. The accepted generalization plan is
> `docs/production_dag_recording_layout_design.md`.

`fisheye.cluster.clipped_inference` composes one immutable LSF workflow for a
clipped recording cohort. It covers detection, detection refinement, finalized
collection creation, flat ROI cache construction, proxy crop binding,
keypoints, keypoint refinement, subject-mask inference, refined subject masks,
content validation, registry reconciliation, and optional NRS cleanup.

This full DAG is the registry-backed default-processing orchestrator, not the
only supported execution surface. Detection-only, cache-only, keypoint-only,
mask-only, refinement-only, recovery, and validation workflows remain
first-class operator entry points. Their stage runners and artifact contracts
are the reusable units composed by this DAG. A user should not need to launch
the complete DAG to run or rerun one stage.

Scheduler packaging is independent of workflow scope. One explicitly selected
work unit may be one ordinary LSF job. Repeated same-resource work should use an
LSF array, and independently safe CPU work may use a bounded in-allocation
bundle. Neither representation makes the underlying stage or standalone job
contract obsolete.

## Composable workflow modules

The detection path is the first extracted domain module. Its raw-detection
fragment accepts a layout-neutral recording target and dispatches either to
the clipped artifact publisher or the whole-video node-local atomic publisher.
The clipped postprocessing builder additionally accepts typed quality-policy,
scheduler, and upstream-gate inputs and returns two things:

- an `LsfWorkflowFragment` containing the detect array, recording-order
  quality-source materialization, collection quality reconciliation,
  detect-refine bundle, and finalized-collection publication;
- typed `DetectionFragmentOutputs` naming every raw/refined group, the quality
  groups, the finalized collection, the terminal LSF job key, and the logical
  artifact key supplied to downstream fragments.

The full planner consumes those outputs when it constructs the cache and mask
commands; it does not independently reconstruct the collection path or
terminal dependency. `compose_detection_workflow(...)` can compose the same
module by itself for a detection-only workflow. This keeps workflow scope
(detection-only versus full analysis) separate from scheduler packaging
(ordinary job, array, or bounded bundle).

For each target, the v2 plan composes native artifact-first detection, legacy
refinement over those same artifact rows, strict recording snapshot
finalization, and five downstream capability fragments:

New plans are emitted as `palette.clipped_inference_bsub_plan.v2`. Validation,
registry reconciliation, cleanup, and the maintained recovery entry points
continue to read v1 plans so completed or interrupted historical campaigns do
not become administratively stranded.

```text
native_detection:<target>
  -> detection_postprocess:<target>
  -> strict_clipped_detection_evidence:<target>
  -> clipped_storage_finalization:<target>
  -> crop_roi_cache:<target>
       ├─ keypoints:<target>
       └─ subject_mask_inference:<target>
              \       /
       subject_mask_refinement:<target>
                 -> analysis_validation:<target>
```

The detector writes each clip once into the selector-free
`detection_artifact_runs` namespace. One recording-level job then assembles
the native canonical `detect_runs` snapshot. Recording quality and legacy
refinement consume the same artifact groups; they do not rerun YOLO or create a
second raw-detection copy. The strict evidence array proves every compatibility
refined clip against the native canonical slice, then publishes the generated
binding, recording refined snapshot, and geometry-only crop-v2 candidate.

`crop_roi_cache` currently provides the stable proxy crop and cache binding,
but it cannot begin until the strict crop-v2 candidate is complete. Keypoints
provides both raw and refined artifacts. Subject-mask
inference does not require keypoints; its refinement fragment joins raw masks
with the exact refined-keypoint output. Cross-recording concurrency gates
remain fragment requirements: a later raw-detection fragment may require the
earlier target's validated-analysis artifact. The campaign finalizer requires
every target validation artifact before registry reconciliation and cleanup.

These logical artifact keys validate composition; concrete LSF dependencies
still enforce execution order. All fragments are resolved into one immutable
workflow before any `bsub` call, so compute jobs never create more scheduler
jobs dynamically. The existing cache, keypoint, and subject-mask stage-only
operator interfaces remain unchanged and are now visible as separate
composition capabilities.

Selected arena geometry and registered gating are available as independent
layout-neutral fragments in `fisheye.cluster.arena_geometry`. They are not yet
silently inserted into this default recipe. A future required-gating policy
must wire the exact `analysis/detection_gate_runs/<run>` output into detection
postprocessing and validate ordered `instance_key` equality before refinement;
optional or absent geometry must remain an explicit workflow policy.

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

All production work is represented as structured `LsfJob` objects. Repeated
same-resource clip work is represented as typed execution tasks inside one LSF
job array, and bounded CPU fan-out is represented as a typed in-allocation
bundle. Every array element and bundle child still receives its own durable
runtime-status envelope, exact command, expected-output checks, and task key.
`--dry-run` writes `plan.json`, `lsf_plan.json`, and per-target detection plans
without calling `bsub`. `--apply` performs the same planning on the workstation
and sends only scheduler-level `bsub` commands through
`login1-citrus-poller` (or
`--submit-host`). No inference, finalization, validation, or Zarr mutation runs
on the workstation or login poller.

Every LSF envelope pins `PYTHONPATH` to the planned cluster-visible
`<repo>/src` for both the runtime wrapper and its child command. This prevents
an editable conda installation from silently importing a different workstation
checkout than the clean `/groups` commit recorded by the plan.
The `run_clipped_inference_dag.sh` planner entry point applies the same pin
before importing the planner itself.

For each recording, the dependency structure is:

```text
artifact detect array[22] -> native recording canonical publication
                          -> sharded recording-order quality source
                          -> collection quality reconcile
                          -> keyed detect-refine CPU bundle[22, max 4]
                          -> finalized compatibility collection
                                      |
                         strict clip evidence array[22]
                                      |
                         generated clipped binding
                                      |
                    recording refined snapshot -> crop-v2
                                      |
                         cache array[6 four-clip bundles]
                                      |
                             proxy crop binding
                               /              \
                    keypoint array[22]   subject-mask array[22]
                              |                  |
                    keypoint finalize           |
                              |                  |
                     keypoint refine            |
                               \                /
                          mask-package array[22]
                                  |
                         refined-mask import
                                  |
                       exact content validation
```

The default limits are eight concurrent detection elements, four concurrent
keypoint elements, four concurrent subject-mask elements, two concurrent cache
elements, four concurrent mask-package elements, and four concurrent
detect-refine bundle children. Keypoint and subject-mask arrays are independent
branches, so up to eight GPU elements for one recording may be active after
proxy creation. These limits are configurable with the corresponding
`--*-concurrency` options and are frozen into the immutable plan.

For a 22-clip recording this is 19 recording-specific `bsub` submissions while
retaining 149 independently identified execution tasks. Registry finalization
and optional NRS cleanup add two campaign-wide submissions. LSF receives array
names in the form `name[1-N]%limit`; array logs contain both `%J` and `%I`.
Downstream stages use a whole-array `done(<job-id>)` barrier, so one failed
element prevents publication. Detection and keypoint node-local scratch paths
also include `LSB_JOBINDEX`, preventing two elements of the same array on one
host from sharing a work directory.

The cache is the physical crop materialization. Each cache array element owns
up to four clips and keeps the cache builder's bounded parallel workers within
one node. The proxy crop runs bind its
rows to the canonical Zarr; there is no redundant standalone crop-image write.
Proxy creation remains one serial recording-level job because the clip proxies
share the same `crop_runs` parent metadata; parallelizing those writes would
need an explicit parent-metadata transaction. Mask-package finalizers remain
array elements rather than a multi-package CPU bundle because each element
already owns eight process workers and its own package artifact.
Keypoints and masks safely read the same immutable cache concurrently. Each
mask package waits for both its clip-local probability shard and the exact
recording-level refined-keypoint run used for left/right eye assignment.

The strict crop-v2 artifact is currently an execution and identity gate for
the compatibility cache path, not yet the cache's direct row-level source.
Binding pixel packages to the crop manifest and its row signatures remains a
separate required gate before the strict candidate set can be atomically
imported or selected.

Detection quality is a recording-level stage, not 22 independent clip-local
state machines. The source materializer streams clip-local raw boxes, stable
`instance_key` values, and canonical parent-frame indices into one immutable
indexed-sharded `detect_collection_sources` run. Parallel quality workers emit
compact traces; one reconciler carries temporal state across every shard and
clip boundary and publishes `detect_quality_runs/<run>`. Each clip refine job
reads only its declared source slice and requires exact key equality before
using those labels. Historical nested `quality_reports` remain a read fallback
for old single-run archives.

`palette.clipped_detect_quality_source.v2` requires full-frame
`source_video_width` and `source_video_height`. The serialized source
materializer resolves those attrs from every selected raw detection run, fails
closed if any run omits them or clips disagree, and stamps the agreed geometry
on the immutable source. Because this job runs after the detection array and
before all downstream work, it is also the single writer that updates compatible
root `width`/`height`, root `source_video_metadata`, and `raw_video` geometry
attrs. Inference dimensions such as `640 x 640` are not full-frame geometry and
must never satisfy this contract.

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

## Detection-quality geometry recovery

Historical `palette.clipped_detect_quality_source.v1` snapshots may be complete
and content-validated while lacking the full-frame geometry introduced by the
v2 contract. Do not rebuild their indexed-sharded arrays or silently relabel
them as v2. `fisheye.utils.repair_clipped_detect_quality_source_geometry`
validates the canonical frame manifest, every raw detection run, complete
source slices, array shapes/dtypes, stored decoded SHA-256 records, and uniform
raw-run geometry before changing metadata. It preserves the v1 `schema_id`,
adds a deterministic `full_frame_geometry_repair` audit record, stamps the
compatible root and `raw_video` geometry, and records that no array payload was
rewritten.

`fisheye.cluster.clipped_inference_detect_quality_recovery` is the one-target
campaign recovery for a geometry failure at this boundary. It fails closed if
the quality source is incomplete or any planned downstream artifact already
exists. The recovery then submits, through the Citrus poller, a short CPU
metadata-repair job followed by a cloned copy of the original DAG from
recording-level detection quality onward:

```text
validate/repair v1 source metadata -> detect quality -> detect refine bundle
  -> finalized detection collection -> ROI cache array -> proxy
  -> keypoint and subject-mask arrays -> finalizers/import
  -> validation -> registry reconciliation -> NRS cleanup
```

The original raw detections, source array objects, artifact run names, model
bindings, and collection lineage remain unchanged. The recovery has its own
immutable plan, logs, statuses, and submission receipt, and runs against an
explicit deployed repository checkout rather than the obsolete source-run
repository snapshot.

If detection quality completed but its scheduler wrapper failed afterward,
`--reuse-complete-quality` accepts it only after checking completion, exact
source binding, row/frame counts, geometry, indexed-sharded array contracts,
and the stored instance-key/trace validation. The continuation begins at the
refine bundle. Recovery also normalizes inherited `short`-queue walltimes to
the queue's one-hour hard limit. Runtime scratch cleanup treats
`LSB_JOBINDEX=0` as a non-array job; positive indices remain isolated array
scratch roots.

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
repair/cleanup -> keypoint array[22] -> keypoint finalize -> keypoint refine
                                                                |
                                      completed raw masks -> mask-package array[22]
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
- [x] A/B the v1 decoded importer and v2 encoded publisher for exact row
  identity, sampled decoded-mask equality, all non-mask arrays, provenance,
  and completion state.
- [x] Validate every boundary row plus deterministic samples from every
  package and channel; verify copied encoded-object digests before decode
  checks.
- [ ] Record wall time, CPU time, peak RSS, bytes read/written, and PRFS object
  operations for both paths.
- [ ] Validate the first full-recording result through Crimson and `MaskStore`.
- [x] Keep v2 behind an explicit feature flag for the first full recording and
  fall back closed to the v1 importer on any contract mismatch.
- [ ] Enable v2 by default only after a full recording passes validation,
  registry reconciliation, Crimson loading, and post-cleanup audit.

Independent of package v2, the v1 importer must pre-index package row
placements once per destination chunk grid. It must not rescan all recording
rows for every destination chunk.

As of 2026-07-15, the deliberately misaligned two-clip PRFS A/B canary passed
for 107,472 rows. The encoded publisher took 58.6 seconds versus 717.8 seconds
for the decoded v1 importer, a 12.25x client-observed speedup. LSF reported
53.5 versus 898.9 CPU seconds, 507 MiB versus 2,778 MiB peak memory, and 78
versus 747 seconds of scheduler runtime. The validator compared all 134
non-mask arrays exactly (80,550,532 values), decoded 2,128 mask rows across 17
deterministic row chunks and all four channels, and included every row of the
single cross-package boundary chunk. The encoded publisher copied 3,356 stored
objects totaling 69,828,072 bytes, verified each destination digest, directly
owned 839 row chunks, and decoded/reassembled only the one boundary row chunk.
The retained report is
`/groups/johnson/johnsonlab/jeremy/palette_smoke/sleepyfish_cam2010093_encoded_import_ab_20260715_v003/validation/ab_report.json`.

The performance-instrumentation item remains open because the canary did not
measure complete filesystem bytes and operation counts for the decoded v1
path. Full-mask parity, Crimson loading, retry accounting, and default rollout
remain gates for the first complete recording.

The canary's standalone v1-to-v2 conversion jobs are compatibility-only. New
clipped workflows must use `--encoded-mask-packages`: the recording-level grid
is frozen before the package array, and each subject-mask finalizer emits its
v2 package directly. This removes old-package extraction and second-package
rewriting from the production DAG. The finalizer still performs the necessary
global-chunk encoding, digest calculation, and one package seal from its fresh
node-local dense output; those package-creation costs are not eliminated. The
production flag remains off by default until the first full recording passes
the remaining gates.

## Sleepyfish pre-array dry run, 2026-07-14

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

This historical dry run predates the array refactor. It rendered 368 LSF
submissions: 122 jobs per target, one all-target registry job, and one NRS
cleanup job. The audit confirmed 66 clips, exact two-way mask
package joins, shared proxy gates for the parallel keypoint/mask branches,
direct cache execution with no nested `bsub`, runtime wrapping on every job,
and no submission evidence. The reviewed snapshot is under
`/tmp/sleepyfish_clipped_full_20260714_v001_dryrun`; it is diagnostic evidence,
not a durable campaign directory.

## Sleepyfish array dry run, 2026-07-15

The post-refactor three-recording dry run covers the same 66 clips and retains
374 task-level status/output contracts, but requires only 44 scheduler
submissions: 14 per recording plus registry finalization and NRS cleanup. It
contains 15 arrays and three bounded CPU bundles. Per recording the array shapes
are detection `[1-22]%8`, cache `[1-6]%2`, keypoints `[1-22]%4`, subject masks
`[1-22]%4`, and mask packages `[1-22]%4`. The detect-refine bundle runs at most
four children in one 16-core, `span[hosts=1]` allocation with four BLAS/OpenMP
threads per child.

The matching one-recording dry run has 16 scheduler submissions and 126 task
envelopes, confirming that one- and multi-recording campaigns use the same DAG
implementation. Neither dry run has an `lsf_submission.json`; no jobs were
submitted. Diagnostic evidence is retained under:

- `/tmp/sleepyfish_lsf_arrays_dryrun_20260715_v003`;
- `/tmp/sleepyfish_lsf_arrays_single_dryrun_20260715_v001`.

The subsequent detection-module extraction was rerun against the same live
registry and recording metadata. Scheduler behavior remained unchanged: the
single-recording plan has 16 submissions, 126 task envelopes, five arrays, and
one bundle; the three-recording plan has 44 submissions, 374 task envelopes,
15 arrays, and three bundles. The immutable plans additionally expose three
logical fragments for one target and seven for three targets. With
`max_active_targets=2`, cam2010096's detection fragment requires cam2010093's
validated-analysis artifact, matching its concrete LSF dependency. No
submission evidence was created. These diagnostic snapshots are retained at:

- `/tmp/sleepyfish_composable_dag_single_dryrun_20260715_v001`;
- `/tmp/sleepyfish_composable_dag_multi_dryrun_20260715_v001`.

## L4 cache-bundle production default, 2026-07-17

Clipped production now defaults to eight clips and eight independent cache
builders per one-L4 bundle. For a 22-clip recording this changes the cache array
from six four-clip elements to three elements (`8 + 8 + 6`) while preserving
one cache artifact and one independently auditable task contract per clip. The
parent cache array concurrency is unchanged; the increased concurrency is
inside each single-GPU allocation.

The default follows a two-order concurrency benchmark and a full eight-clip
cam2010096 canary. The full canary produced 112.5 GB across 429,077 rows in
13.6 minutes, with 92% median NVDEC utilization, no swap, and successful
validation and cleanup of all child caches. Shared LSF GPU process mode remains
required; exclusive-process mode cannot host the independent decoder children.
