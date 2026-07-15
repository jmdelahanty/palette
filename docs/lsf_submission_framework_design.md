# LSF Submission Architecture: Shared Kernel and Job-Family Planners

- Status: Phase 1 and the Phase 2 dry-run implementation are complete; no
  GoodCopBadCop jobs have been submitted
- Last reviewed: 2026-07-10
- Scope: Palette commands that plan or submit LSF jobs with `bsub`

## Purpose

Palette has several strong LSF workflows, but their submission guarantees are
not yet consistent. The clipped-collection keypoint workflow is the clearest
current reference: it builds a deterministic plan, assigns explicit output run
names, submits jobs with real `done(<jobid>)` dependencies, records each
submission incrementally, and never asks a downstream stage to guess which
upstream run it should consume.

This document proposes applying those practices across Palette without creating
one universal pipeline abstraction. The intended architecture is:

```text
family planner                         shared LSF kernel
-------------------------------        -------------------------------
detect targets and run names     --->  job/resource/dependency models
keypoint targets and run names   --->  plan and submission bundles
mask targets and run names       --->  bsub rendering and job-id parsing
cache targets and publication    --->  atomic submission snapshots
import/analytics commands        --->  job runtime/status envelope
```

The shared layer owns cluster orchestration mechanics. Each job family owns its
scientific and storage semantics.

## Decision

Use a small Python LSF orchestration kernel with separate family-specific
planners. Do not build a generic YAML workflow language, a stage plugin
registry, or a single command that attempts to understand every Palette stage.

The boundary is deliberate:

- The shared kernel knows what a job is, what resources it requests, which
  other jobs it depends on, how it is submitted, and where its evidence lives.
- A detect planner knows detect models, artifact import, quality reports, and
  refined-detect lineage.
- A keypoint planner knows crop/cache inputs, pose schemas, pose models,
  keypoint run names, and refinement.
- A subject-mask planner knows components, assignment keypoints, dense output
  authority, shard/finalizer behavior, and derived caches.
- A crop/cache planner knows crop signatures, cache aliases, publication, and
  node-scratch staging.
- Import, review-proxy, analytics, and training planners retain their own input
  validation and completion contracts.

Family planners may share family-local helpers. For example, whole-recording
and clipped-collection keypoint planners should share keypoint command and run
name construction without forcing those details into the LSF kernel.

Stage-scoped planners are permanent first-class interfaces, not transitional
wrappers around a monolithic DAG. A full analysis DAG exists to automate the
usual registry-backed dependency chain over a dataset. Operators must still be
able to request keypoints, masks, detections, refinement, validation, or repair
independently. The planner chooses ordinary jobs, arrays, or bounded bundles
according to work cardinality and resource ownership; those are scheduler
representations, not different scientific workflows.

### Implementation status

The Phase 1 extraction and the first Phase 2 keypoint-family slice were
implemented on 2026-07-10:

- `fisheye.cluster.lsf.models` defines immutable resource, dependency, and job
  models;
- `fisheye.cluster.lsf.backend` owns `bsub` argument rendering, structured
  `done`/`ended` dependencies, job-ID parsing, placeholder resolution, command
  execution, and retained diagnostics;
- `fisheye.cluster.lsf.bundle` owns atomic JSON snapshot persistence;
- `fisheye.cluster.lsf.models.LsfWorkflow` validates unique identities,
  dependency references, cycles, and deterministic topological order;
- `fisheye.cluster.lsf.submission` submits that graph, retains every `bsub`
  result, and atomically records accepted jobs or partial submission failure;
- the clipped-collection keypoint planner now renders shard, finalizer, and
  refinement submissions through those shared primitives and delegates its LSF
  apply loop to the common topological submitter;
- its existing plan/submission schemas, commands, dependency expressions, and
  private compatibility imports used by clipped subject masks are preserved;
- it additionally emits common `lsf_plan.json` and `lsf_submission.json`
  evidence without replacing the family-specific artifacts;
- focused shared-kernel, clipped-keypoint, and clipped-subject-mask tests cover
  the compatibility boundary.
- `fisheye.cluster.lsf.runtime` now owns the common per-job runtime envelope,
  LSF token expansion, signal forwarding, atomic running/final status,
  expected-output checks, and fail-closed job-local scratch cleanup;
- `LsfExecutionGroup` and `LsfExecutionTask` now represent repeated work as
  either an LSF array or a bounded in-allocation bundle. The backend renders
  `name[1-N]%limit`, while `fisheye.cluster.lsf.task_group` selects array
  elements from the immutable plan or runs bundle children with a fixed
  concurrency limit. Every child retains its own command, status, cleanup,
  and expected-output contract;
- `LsfWorkflowFragment` and `compose_lsf_workflow` validate logical artifact
  requirements across independently built subgraphs. The clipped full-analysis
  planner now uses this contract rather than merely grouping jobs after the
  fact;
- `fisheye.cluster.clipped_detection` is the first extracted clipped domain
  module. It returns both its LSF fragment and typed outputs for raw groups,
  recording-quality groups, refined groups, finalized collection, terminal
  job, and logical artifact. The same builder composes into the full campaign
  or a detection-only workflow, while `fisheye.cluster.clipped_lsf` holds the
  shared clipped runtime-envelope construction;
- `fisheye.cluster.keypoints.common` owns exact pose-model, flat-cache, run-name,
  live crop-DAG capability, prediction-job, and refinement-job bindings used
  within the keypoint family;
- the clipped implementation now lives in
  `fisheye.cluster.keypoints.clipped_collection`; the historical
  `fisheye.utils.plan_clipped_collection_keypoints_bsub` module is a thin
  compatibility entry point for existing scripts and subject-mask imports;
- `fisheye.cluster.keypoints.whole_recording` owns the explicit target manifest,
  target/model/cache preflight, deterministic output collision checks, one
  prediction-to-refinement chain per recording, and registry-finalizer fan-in;
- `fisheye.cluster.keypoints.registry_finalize` validates the exact keypoint,
  refined-keypoint, crop, and model lineage before serially refreshing
  keypoint performance and both registry step-status rows;
- `run_keypoints_with_registry_model` accepts an exact `--model-run-id`, and
  YOLO keypoint status writes now honor `PALETTE_DISABLE_REGISTRY_WRITES` so the
  family planner can defer SQLite writes safely;
- `scripts/submit_whole_recording_keypoints_bsub.sh` is the operator-facing
  wrapper. It requires an explicit `--dry-run` or `--apply`; only apply calls
  the shared `bsub` submitter.
- ordinary-batch, clipped-collection, and whole-recording keypoint submitters
  expose the same default/custom/opt-out storage controls. Plans record both
  `keypoint_storage.requested` and `keypoint_storage.effective`; immutable YOLO
  outputs default to 262,144-row ROI shards and 262,144-row frame shards, while
  `--no-keypoint-sharding` remains an explicit compatibility override.

Recovery behavior, a cross-family conformance scanner, and broad migration of
detect/mask/cache submitters remain later slices. The new whole-recording
planner has been validated locally with fake LSF submissions, but it has not
been run against the GoodCopBadCop target set or submitted on Citrus Poller.

## Current Repository Inventory

At the start of this extraction on 2026-07-10, Palette had 32 named LSF
planning/submission surfaces:

- 28 shell wrappers or submitters under `scripts/`;
- 4 Python planners/submitters under `src/fisheye/utils/`.

The count includes thin shell wrappers around Python planners because those are
operator-facing submission surfaces. The implementations fall into three broad
tiers.

### Tier A: structured DAG planners

The strongest current references are:

- `fisheye.utils.plan_clipped_collection_keypoints_bsub`;
- `fisheye.utils.plan_clipped_collection_subject_masks_bsub`;
- `fisheye.utils.submit_clipped_detect_refine_plan_bsub`;
- `fisheye.utils.submit_review_proxy_videos_sharded_bsub`.

Together, these demonstrate most of the desired behavior:

- dry-run planning before submission;
- deterministic work-unit and output names;
- structured JSON plans and/or submission manifests;
- explicit job resources and log paths;
- parsed LSF job IDs rather than dependency-by-name guessing;
- fan-out, linear chains, and fan-in dependencies;
- explicit downstream run names;
- target collision preflight;
- incremental submission snapshots;
- generated per-stage scripts with `set -euo pipefail`;
- unit tests using fake command runners;
- finalizers that publish shared metadata only after workers succeed.

No one implementation currently contains every desirable feature. The clipped
keypoint planner has the cleanest plan/submission separation and placeholder
resolution. The clipped detect/refine submitter has the strongest per-stage
status evidence and output-collision preflight. The sharded review-proxy
submitter cleanly separates shard production from one authoritative manifest
publisher.

### Tier B: composed shell dependency chains

Several shell submitters implement useful dependency patterns directly:

- `submit_detect_artifact_quality_refine_bsub.sh` submits one deterministic
  prediction/postprocess chain per target.
- `submit_subject_mask_batches_bsub.sh` submits a GPU inference array followed
  by a dependent CPU finalization array.
- `submit_crop_flat_roi_cache_bsub.sh` submits crop followed by cache creation.
- `submit_crop_flat_roi_cache_batches_bsub.sh` fans many cache jobs into one
  serial registry finalizer.
- `submit_refine_keypoints_batches_bsub.sh` supports an upstream dependency and
  a dependent registry finalizer.
- `submit_detect_quality_refine_bsub.sh` demonstrates a simple linear
  detect-to-quality-to-refine chain, while documenting why selecting `latest`
  makes it inappropriate for concurrent production use.

These scripts prove the execution patterns, but independently reimplement job
ID parsing, command rendering, dependency construction, dry-run behavior, and
submission logging.

### Tier C: independent arrays and one-off jobs

The remaining submitters cover discovery, arrays, imports, diagnostics,
benchmarks, analytics, and single-session jobs. Most create useful run
directories and logs, but they do not all emit the same plan, submission, and
runtime evidence. Several also submit by default with `--dry-run` as an opt-out,
while newer Python DAG tools plan by default and require an explicit apply or
submit flag.

### Existing shared code is too small

`scripts/lib/palette_lsf.sh` currently provides shell command printing, `bsub`
execution, and job-ID extraction. The Python planners each have similar local
versions of:

- resource argument construction;
- shell quoting;
- job-ID parsing;
- placeholder replacement;
- command execution;
- plan serialization;
- submission snapshot writing.

Those mechanics are the right extraction target. Stage discovery and command
construction are not.

### Family ownership map

The migration unit should be a job family, not an individual shell script and
not the entire repository at once:

| Family | Current operator surfaces | Family convergence target |
| --- | --- | --- |
| Detect | detect batches, artifacts, quality/refine chains, compute smokes, decode parity, clipped detect/refine | Deterministic prediction/artifact/import/quality/refine planners with explicit runs |
| Keypoints | keypoint batches, refinement batches, clipped keypoints | Whole-recording and clipped planners sharing keypoint-specific work-unit builders |
| Subject masks | subject-mask batches/finalization, clipped subject masks, SAM, finalizer diagnostics | Component-aware inference/finalization planners preserving dense authority and assignment lineage |
| Crop and ROI cache | crop batches, flat-cache jobs and bundles, clipped cache bundles | Crop/cache publication planners with scratch staging, validation, and registry fan-in |
| Review proxies | single and sharded review-proxy submitters | Shard/finalizer planner with one authoritative manifest publisher |
| Imports | Citrus session import and recording/training import | Small import planners with input contracts and common runtime evidence |
| Analytics and training | chaser analytics and training review bootstrap | Family-owned single jobs or short chains using the common evidence envelope |
| Diagnostics and benchmarks | detect parity/smokes and subject-mask finalizer benchmarks | Explicitly non-production family tools using the same safe submission mechanics |

Thin shell commands may remain as stable operator entry points. They should
delegate planning and submission to their family planner rather than each
reimplementing LSF behavior.

## Dependency Patterns to Preserve

### Linear chain

```text
predict -> validate -> quality -> refine -> validate
```

Each job depends on `done(<previous-job-id>)`. A failed stage prevents later
mutation. This is appropriate when every stage has one exact predecessor.

### Fan-out and fan-in

```text
shard A --\
shard B ----> finalizer -> validation or refinement
shard C --/
```

The finalizer uses:

```text
done(<job-A>) && done(<job-B>) && done(<job-C>)
```

Workers must not publish shared selectors or authoritative collection metadata.
The finalizer validates all expected outputs and publishes once.

### Per-target chains

```text
target A: GPU prediction A -> CPU refinement A
target B: GPU prediction B -> CPU refinement B
target C: GPU prediction C -> CPU refinement C
```

This isolates failures and allows successful targets to continue. It is the
preferred shape when downstream stages must consume a target-specific run or
artifact.

### Array barrier

```text
GPU array -> CPU array
```

An array barrier is compact and gives a natural `%<max-active>` throttle, but
`done(<array-job-id>)` is a whole-array success gate. One failed element blocks
all downstream elements. Use it only when that behavior is intended and both
arrays consume the same immutable index manifest.

Palette should not assume per-index array dependency behavior until the exact
LSF syntax and semantics are verified on the Janelia cluster and covered by a
cluster smoke.

### `done` versus `ended`

Production dependencies should default to `done`, meaning the upstream job
completed successfully. `ended` permits downstream execution after an upstream
failure and is suitable only for an explicitly named cleanup, diagnostics, or
recovery job. A family planner must opt into it visibly; callers should not pass
unreviewed raw dependency strings.

## Required Submission Contract

Every new or migrated LSF workflow should produce one submission bundle. A
suggested layout is:

```text
<run-dir>/
├── plan.json
├── submission.json
├── targets.jsonl                 # or a typed TSV when streaming is useful
├── scripts/
│   ├── <job-key>.sh
│   └── ...
├── logs/
│   ├── <job-key>.%J.out
│   ├── <array-key>.%J.%I.out
│   └── <job-key>.%J.err
├── status/
│   ├── <job-key>.<job-id>.json
│   ├── <array-task>.<job-id>.<array-index>.json
│   ├── <bundle-task>.<job-id>.json
│   ├── <bundle-key>.<job-id>.bundle.json
│   └── ...
└── progress/                     # optional family-specific live progress
```

### `plan.json`

The plan is immutable evidence of intent. It should be complete before the
first `bsub` call and contain:

- a versioned schema identifier;
- workflow family and workflow/run ID;
- creation timestamp, repository path, Git commit, and planner invocation;
- typed work units and stable work-unit keys;
- exact input paths and input run IDs;
- deterministic output run names and artifact paths;
- commands as argument arrays, not only display strings;
- resource requests, queues, and walltimes;
- dependency references by stable job key;
- typed array/bundle execution groups with stable task keys, commands,
  concurrency limits, and per-task output contracts;
- stdout, stderr, status, and progress paths;
- collision/precondition results;
- hashes for target manifests and important immutable inputs when practical;
- registry-write policy and any finalizer job.

A plan must not contain real LSF job IDs because they do not exist yet. It may
contain structured references such as `{"job_key": "keypoints:recording-A"}`.

### `submission.json`

The submission snapshot records what was actually sent to LSF. It should be
written atomically before submission begins and after every successful `bsub`
call. It should contain:

- `planned`, `submitting`, `submitted`, or `submission_failed` submission
  state;
- the plan path and plan digest;
- each job key, real job ID, rendered dependency, and exact submitted command;
- raw `bsub` stdout/stderr;
- resolved log and status paths;
- partial results and the exception when orchestration stops midway.

`submitted` means LSF accepted the jobs. It must not be used to mean the jobs
ran successfully.

### Per-job runtime status

Every generated job script should write a small atomic status document with:

- schema, workflow ID, job key, work-unit key, and stage;
- LSF job ID/index/name/queue and execution host;
- start and finish timestamps;
- exact command and relevant environment selections;
- `running`, `succeeded`, or `failed` execution state;
- process return code and concise error context;
- expected outputs and validation results.

The runtime envelope should install traps so failure evidence is written even
when the stage command exits nonzero. Stage tools may continue to write richer
reports; the shared status is only the common operational envelope.

### Completion checks

A generic checker can report submission state, LSF identity, log presence, and
runtime status. Family-specific checkers remain responsible for semantic
completion, such as:

- whether a refined keypoint run has the expected source run and row count;
- whether dense `masks_roi` is physically present;
- whether a cache manifest and payload agree;
- whether an imported run validates against its artifact contract.

This separation prevents the generic layer from becoming a second, incomplete
implementation of every stage validator.

## Shared LSF Kernel

The first reusable implementation should be a small Python package, for
example:

```text
src/fisheye/cluster/lsf/
├── models.py       # immutable JSON-ready plan models
├── backend.py      # render/submit bsub and parse job IDs
├── bundle.py       # atomic plan/submission persistence
├── runtime.py      # generated job status envelope
└── inspect.py      # common operational status readout
```

The exact module names can change, but responsibility should remain narrow.

### Core models

The minimum useful models are:

```text
ResourceSpec
  queue, ncores, mem_gb, walltime, gpus, extra_lsf_args

CommandSpec
  argv, cwd, environment, display_name

DependencySpec
  upstream_job_keys, condition=all_succeeded|all_ended

JobSpec
  job_key, work_unit_key, stage, resources, commands,
  dependency, log paths, status path, expected outputs

ExecutionTask
  task_key, stage, command, status path, cleanup paths, expected outputs

ExecutionGroup
  mode=array|bundle, tasks, max_concurrent, optional bundle summary path

WorkflowPlan
  schema, family, workflow_id, jobs, targets, preflight, metadata
```

Dependencies should reference job keys. Only the LSF backend translates those
references into expressions containing real job IDs.

### Kernel responsibilities

The shared implementation should:

1. validate unique job keys and an acyclic dependency graph;
2. verify every dependency reference resolves;
3. topologically order submissions;
4. render shell-safe `bsub` argument arrays;
5. parse job IDs and fail closed when parsing fails;
6. update `submission.json` atomically after each submission;
7. generate or invoke the common runtime status envelope;
8. support plan-only mode without `bsub` in `PATH`;
9. preserve enough raw evidence to reproduce or diagnose every submission;
10. expose fake-runner seams for unit tests.
11. render array indices and slot limits without expanding them into repeated
    `bsub` calls;
12. preserve one status/output-validation envelope per array element or bundle
    child;
13. require array scratch namespaces to include both `LSB_JOBID` and
    `LSB_JOBINDEX`.

### Kernel non-responsibilities

The shared implementation should not:

- discover recordings or choose which datasets are eligible;
- resolve models or pose schemas;
- choose source runs via `latest`;
- define detect, keypoint, mask, cache, or import commands;
- decide whether outputs are scientifically valid;
- own Zarr mutations or registry reconciliation;
- invent retries, delete outputs, cancel jobs, or resume partial workflows;
- abstract LSF into a hypothetical multi-scheduler API before another backend
  is actually required.

## Family-Specific Planners

### Detect

The detect family should own:

- model resolution or explicit model pinning;
- deterministic detect, quality, and refined run names;
- direct-write versus artifact/import topology;
- decode backend and resize contract;
- detect-quality parameters;
- output collision preflight;
- detect and refined-detect validators.

The production default should follow the deterministic artifact path rather
than a downstream `latest` selector.

### Keypoints

The keypoint family should own:

- source crop run and crop revision;
- one cache manifest per target and cache validation;
- model set/run resolution and pose schema compatibility;
- inference parameters and input mode;
- deterministic keypoint and refined-keypoint run names;
- exact source-run handoff to refinement;
- keypoint/refinement validation and review policy;
- deferred registry reconciliation.

Whole-recording and clipped-collection workflows can use different planners
that share these constructors and validators.

The intended keypoint-family layout is conceptually:

```text
keypoints/
├── common.py               # keypoint-only builders and validators
├── clipped_collection.py   # clipped proxy/shard/finalize/refine DAG
└── whole_recording.py      # recording prediction/refine/registry DAG
```

The clipped and whole-recording implementations now occupy those separate
modules. Existing `fisheye.utils.*` imports remain stable through a thin
compatibility entry point, with focused compatibility tests. The separation is
semantic as well as organizational: clipped proxy/shard/collection finalizing
does not appear in the whole-recording planner.

### Subject masks

The subject-mask family should own:

- component selection and source crop/cache mapping;
- assignment-keypoint lineage;
- raw shard and refined output names;
- dense `masks_roi` authority;
- finalization, derived cache refresh, and component validation;
- safe chunk-aligned write topology;
- registry and component-quality finalization.

Legacy eye-mask compatibility must not become the primary abstraction.

### Crop and ROI cache

The crop/cache family should own:

- crop signature and refined-detect lineage;
- cache key, alias, manifest, payload, and row-sidecar paths;
- node-local scratch staging and cleanup;
- payload-first/manifest-last publication;
- cache validation and overwrite policy;
- registry finalization after fan-out jobs succeed.

### Imports, analytics, review proxies, and training

These families may use a single job or a small chain. They should adopt the
same plan/submission/runtime evidence without pretending to be detect or
keypoint workflows. The framework must remain worthwhile for a one-job Citrus
import as well as a many-job DAG.

## GoodCopBadCop Keypoint and Refinement Example

The GoodCopBadCop rerun is a good first whole-recording keypoint pilot because
it requires target-specific cache manifests and exact prediction-to-refinement
lineage.

### Preflight findings

The current filesystem/registry inspection found 40 GoodCopBadCop recording
directories in the intended recordings scope, while current registry-backed
keypoint discovery emits 31 analysis Zarrs. The difference includes:

- four 2026-05-29 recordings;
- four 2026-07-02 recordings;
- the root recording copy of
  `2026-06-14T21-12-08Z_arena_4_GoodCopBadCop`, whose registry analysis dataset
  currently points to a separate heartrate example path.

Therefore the first plan must use an explicit, reviewed target manifest rather
than treating `--path-contains GoodCopBadCop` as authoritative.

### Planned work unit

Each recording work unit should resolve and record:

```text
recording_id
recording_dir
analysis_zarr
source_crop_run and crop revision
selected pixel source and rejected alternatives
flat ROI cache manifest, payload, key, and validation result
pose model set, model run, model path, and digest
pose schema
keypoint run name
refined-keypoint run name
prediction and refinement resources
expected output group paths
```

The previous production-compatible settings provide a starting point, not an
implicit default:

- model set `pose_all_registry_reviewed_v2_keypoints_20260520_v001`;
- model run
  `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2`;
- pose schema `traditional_v2`;
- minimum zebrafish inference surface `348x348`;
- `gpu_l4`, one GPU, four CPU slots for inference;
- inference batch size 256;
- flat ROI cache staged to node-local scratch;
- CPU `short` queue for refinement.

The planner must resolve these again and store the result in `plan.json` before
submission.

### Explicit target manifest

The planner accepts only a reviewed JSON manifest. A minimal two-recording
example is:

```json
{
  "schema": "palette.whole_recording_keypoint_targets.v1",
  "expected_target_count": 2,
  "targets": [
    {
      "target_id": "2026-06-14_arena_1",
      "recording_id": "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
      "recording_dir": "/groups/.../2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
      "analysis_zarr": "/groups/.../zarr/recording_analysis.zarr",
      "roi_cache_manifest": "/groups/.../cache.flat_roi_cache.json",
      "crop_run": "crop_20260614"
    },
    {
      "target_id": "2026-06-14_arena_2",
      "recording_id": "2026-06-14T21-12-08Z_arena_2_GoodCopBadCop",
      "recording_dir": "/groups/.../2026-06-14T21-12-08Z_arena_2_GoodCopBadCop",
      "analysis_zarr": "/groups/.../zarr/recording_analysis.zarr",
      "roi_cache_manifest": "/groups/.../cache.flat_roi_cache.json",
      "crop_run": "crop_20260614"
    }
  ]
}
```

Relative paths are resolved against the manifest directory. Preflight requires
unique target, recording, and Zarr identities; an analysis Zarr registered to
that recording; a complete flat cache whose archive/crop/signature/revision and
payload size match; a live, complete crop DAG node with matching signature,
revision, row count, and ROI shape; a cache and live crop surface of at least
`348x348`; one exact successful model artifact with a registry digest; and
absent deterministic output groups. The plan records whether persisted
`roi_images` and the acquisition/derived crop video are independently eligible.
Derived-video eligibility uses `ffprobe` dimensions and requires them to match
the live crop node as well as meet the minimum size.
For GoodCopBadCop the small persisted crop images are rejected, while a valid
`348x348` or larger flat cache is sufficient even if the derived crop video is
missing or unavailable at execution time.

### Login-node dry run

From `login1-citrus-poller`, use the groups checkout and run the planner itself
on the login node. Do not wrap the planner in another `bsub`: in apply mode it
is the submit-side process that issues and records every individual `bsub`.

```bash
cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette

scripts/submit_whole_recording_keypoints_bsub.sh \
  --manifest /groups/johnson/johnsonlab/jeremy/manifests/goodcopbadcop_keypoints_20260710.json \
  --run-label goodcopbadcop_kpt5_traditional_v2_20260710_v001 \
  --run-root /groups/johnson/johnsonlab/jeremy/logs/whole_recording_keypoints/goodcopbadcop_kpt5_traditional_v2_20260710_v001 \
  --model-set-id pose_all_registry_reviewed_v2_keypoints_20260520_v001 \
  --model-run-id pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2 \
  --min-roi-size 348 \
  --dry-run
```

This performs read-only target/model/cache/output preflight, writes `plan.json`,
`targets.normalized.json`, `zarr_paths.txt`, and `lsf_plan.json`, and prints all
`bsub` templates. It does not call `bsub`. The plan should be reviewed before
the same invocation is changed to `--apply`; that apply step is intentionally
not authorized or performed as part of the current work.

On apply, each accepted job is recorded immediately in
`lsf_submission.json`. Worker status goes under `status/`, prediction progress
under `progress/`, LSF stdout/stderr under `logs/`, and the serial reconciliation
report under `registry/`.

### Recommended DAG

```text
keypoints:recording-01 -> refine-keypoints:recording-01 --\
keypoints:recording-02 -> refine-keypoints:recording-02 ----> registry finalizer
keypoints:recording-03 -> refine-keypoints:recording-03 --/
```

Use one prediction job and one dependent refinement job per recording. This
matches the clipped planner's failure isolation and explicit lineage:

- refinement starts only after that recording's prediction succeeds;
- one failed prediction does not prevent other recordings from refining;
- refinement receives `--keypoint-run <planned-name>`;
- no stage selects `latest`;
- GPU allocation ends before CPU refinement begins;
- the registry finalizer waits for every planned chain to succeed and writes
  shared registry state serially.

If the final registry operation requires every target to succeed, depend on all
refinement jobs with `all_succeeded`. If partial reconciliation is desired, it
must be a separately designed recovery/finalization mode that reads per-target
status; it should not silently replace `done` with `ended`.

### Why the two existing batch submitters should not simply be chained

The current keypoint batch job catches individual recording failures and
continues. LSF can therefore mark an array element successful even when one of
the recordings inside it failed. A dependent refinement array that selects
`latest` could then refine an older run.

The current refinement submitter accepts one upstream job dependency, but its
default keypoint source is `latest`. It becomes safe only when the keypoint run
is explicit and common across every target, every prediction failure reaches
LSF as a nonzero exit, and the target manifest is identical across stages.

The family planner should fix the topology and lineage instead of adding more
shell conventions around these hazards.

## Registry Write Policy

SQLite registry writes should not be fanned out across many workers when they
can be derived from completed Zarr outputs afterward.

The preferred pattern is:

1. worker jobs set `PALETTE_DISABLE_REGISTRY_WRITES=1` where the stage supports
   it;
2. workers write complete stage outputs and per-job evidence;
3. a serial finalizer validates planned outputs;
4. the finalizer reconciles registry status and performance rows;
5. the finalizer records exactly which work units were reconciled.

Family planners own the reconciliation command because registry projections are
stage-specific. The shared kernel only schedules it and records its evidence.

## Safety and Operator Contract

New family planners should follow these CLI rules:

- planning is the default, or `--dry-run` and `--apply` are mutually exclusive
  with no ambiguous mode;
- submission requires an explicit `--apply` or `--submit`;
- broad application can require `--allow-multiple` until the family workflow
  has passed limited cluster smokes;
- a stable `--workflow-id` or `--run-id` is accepted and included in every
  output name;
- an existing run directory or planned output collision fails closed;
- overwrite is explicit and family-specific;
- the exact submit-host repository path and Git commit are recorded;
- commands use `scripts/py` for Python;
- all generated commands and dependency expressions are printed in plan mode;
- submission refuses to proceed if `bsub` is absent;
- submission never depends on parsing human-readable output without retaining
  that raw output in the bundle.

## Conformance and Enforcement

Not every workflow needs a DAG, but every workflow needs the same minimum
submission evidence. Define two conformance levels:

### Single-job conformance

Required even for imports, diagnostics, and one-off analytics:

- plan-first or explicit dry-run behavior;
- stable job/workflow identity;
- exact command and resources;
- durable `bsub` response and parsed job ID;
- deterministic stdout, stderr, and runtime status paths;
- atomic submission state;
- fail-closed output collision policy where the job mutates durable data.

### DAG conformance

Required in addition when a workflow has dependencies or shared publication:

- structured dependency references;
- explicit upstream run/artifact lineage;
- incremental snapshots after each submission;
- per-stage runtime evidence;
- fan-in finalizers for shared state;
- family semantic completion checks.

After the shared backend exists, add an architectural test that inventories
submission surfaces and rejects new direct `bsub` execution or new job-ID
parsers outside:

- the shared LSF backend;
- thin operator wrappers that delegate to it;
- a versioned legacy exception list with an owner and migration phase.

The check should report new surfaces for review rather than attempt to infer
scientific correctness. Existing scripts can remain on the exception list
during migration, but the list should only shrink. This gives the repository a
practical ratchet without requiring an all-at-once rewrite.

## Testing Strategy

### Shared kernel tests

- JSON round-trip for every model;
- unique job-key and dependency-reference validation;
- cycle detection;
- deterministic topological ordering;
- `done` and explicitly opted-in `ended` rendering;
- shell-safe argv and resource rendering;
- job-ID parsing success and failure;
- atomic incremental snapshots under partial submission failure;
- fake-runner submission with realistic `bsub` output;
- generated runtime wrapper success, command failure, and signal behavior.

### Family planner tests

- deterministic target and output names;
- exact upstream-to-downstream run lineage;
- missing input and output-collision preflight;
- resource selection by stage;
- correct cache/model/component options;
- semantic validator and finalizer commands;
- plan-only behavior with no LSF installation.

### Cluster smoke sequence

For each migrated family:

1. plan one work unit;
2. submit one work unit without a downstream fan-in;
3. submit two work units plus a finalizer to test dependencies;
4. verify failure of an upstream job blocks only its intended downstream jobs;
5. verify status and log paths from the bundle;
6. run the family completion checker;
7. only then allow broad submission.

## Migration Plan

### Phase 0: adopt the contract for new work

- Treat Tier A planners as the reference for new submission code.
- Require explicit run lineage and a plan artifact in new multi-stage jobs.
- Avoid adding new direct `bsub` parsing implementations.

### Phase 1: extract the shared kernel without behavior changes

- Extract tested job-ID parsing, resource rendering, dependency resolution,
  command running, and atomic submission snapshots from the clipped planners.
- Preserve their existing plan schemas and commands initially through adapter
  functions.
- Move shell-only helpers only when a migrated caller needs them; do not rewrite
  all scripts at once.

### Phase 2: add the whole-recording keypoint family planner

- Build an explicit-target keypoint prediction/refinement planner.
- Use GoodCopBadCop as the first dry-run and limited apply workflow.
- Add target/cache/model preflight and deterministic run names.
- Add per-recording prediction-to-refinement dependencies and serial registry
  reconciliation.
- Keep the existing batch submitters available during validation.

### Phase 3: converge detect planners

- Move the deterministic artifact/import/quality/refine path onto the shared
  kernel.
- Retain direct-write detect submission for constrained smokes, clearly labeled
  as such.
- Reuse the existing clipped detect/refine completion checker concepts.

### Phase 4: converge subject-mask and crop/cache planners

- Migrate split GPU/CPU subject-mask workflows while preserving dense-mask and
  chunk-alignment contracts.
- Migrate crop/cache publication and registry fan-in.
- Preserve family-specific scratch, package, and validation behavior.

### Phase 5: adopt the common evidence envelope in simple submitters

- Use the shared bundle and runtime status for imports, analytics, training,
  diagnostics, and one-off jobs.
- Do not force single jobs to manufacture meaningless DAG or artifact concepts.
- Deprecate old wrappers only after their replacement has equivalent dry-run,
  resource, logging, and recovery behavior.

## Acceptance Criteria

The architecture is successful when:

- every active submission surface can emit a complete plan before calling
  `bsub`;
- every accepted LSF job has a durable job key, job ID, exact command, resource
  request, dependency, and log/status path;
- downstream jobs consume explicit upstream runs or artifacts;
- a partial submission can be diagnosed without reconstructing terminal
  output;
- worker failures are visible to LSF and block the intended dependents;
- shared selectors, manifests, and registry state are published by validated
  finalizers where concurrency requires it;
- families share orchestration code without sharing stage semantics;
- existing family validators remain the authority for scientific completion.

## Open Implementation Questions

These should be answered during the Phase 1 prototype:

1. Should the common runtime envelope execute one argv command or a short
   ordered list of argv commands? The latter covers import/validate chains while
   still avoiding arbitrary shell programs.
2. Should individual jobs be concurrency-limited through an LSF job-group
   feature, or should families choose between individual jobs and arrays? Do not
   adopt job groups until their behavior is verified on the cluster.
3. Which existing plan schemas should be preserved indefinitely, and which can
   receive a versioned conversion to the common `WorkflowPlan` schema?
4. Should common operational inspection query LSF directly, or initially rely
   only on submission and runtime evidence? Direct LSF history queries may be a
   later, optional adapter.
5. The whole-recording keypoint pilot now uses
   `fisheye.cluster.keypoints.registry_finalize` as its canonical combined
   prediction/refinement reconciliation command. Other keypoint topologies may
   reuse its exact-run validation or provide a topology-specific finalizer.
6. Should a recovery command submit only missing jobs from an existing plan?
   Recovery is valuable, but it should follow completion/status contracts and
   must not be mixed into the first extraction.

## Next Implementation Slice

The shared kernel, runtime envelope, keypoint-family builders, separate clipped
module, and whole-recording planner are now implemented. The next lowest-risk
slice is operational validation, not another abstraction:

1. build and review the explicit 40-recording GoodCopBadCop manifest, resolving
   the nine filesystem/registry discrepancies and binding one cache per target;
2. run only the documented `--dry-run` on `login1-citrus-poller` and inspect the
   generated plan and command templates;
3. after separate authorization, validate one target end to end, including the
   runtime and serial registry-finalizer evidence;
4. after that succeeds, validate two independent target chains and their fan-in
   dependency before any broad submission;
5. add a common inspection/recovery command, then begin a family-by-family
   detect migration without changing detect semantics.

The clipped-collection planner remains a separate family module throughout
this work; it is the compatibility reference, not the universal keypoint
planner.
