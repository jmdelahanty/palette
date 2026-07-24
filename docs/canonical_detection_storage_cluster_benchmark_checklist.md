# Canonical Detection Storage Cluster Benchmark Checklist

Status: active implementation checklist

Date established: 2026-07-24

Related contracts:

- [`canonical_detection_storage_implementation_checklist.md`](canonical_detection_storage_implementation_checklist.md)
- [`shared_zarr_storage_benchmark_contract.md`](shared_zarr_storage_benchmark_contract.md)
- [`lsf_submission_framework_design.md`](lsf_submission_framework_design.md)
- [`dask_zarr_write_safety.md`](dask_zarr_write_safety.md)
- [`zarr_storage_lifecycle_policy.md`](zarr_storage_lifecycle_policy.md)

## Goal

Build a resumable LSF benchmark workflow that selects an evidence-backed
storage profile for canonical detections. Every measured candidate must follow
the intended production lifecycle:

```text
noncanonical safe copy on shared storage
    -> stage once to job-local scratch
    -> validate and compute only from the local copy
    -> publish to a fresh shared benchmark destination
    -> validate the published result
    -> run published-reader and Crimson checks
```

The workflow measures storage policy without modifying a canonical analysis or
training Zarr, updating selectors, or registering benchmark outputs.

## Locked Decisions

- [x] Use only a noncanonical, selector-ineligible source copy for cluster
      benchmarks.
- [x] Stage the source copy to node-local scratch before canonical conversion or
      candidate computation.
- [x] Exclude shared-source reads from candidate compute timing.
- [x] Build every candidate from the same validated local canonical staging
      store.
- [x] Measure local materialization separately from publication to shared
      storage.
- [x] Publish only to a dedicated, fresh `.palette_benchmarks` namespace.
- [x] Permit PRFS reads only after publication, as explicit published-reader
      workloads rather than compute inputs.
- [x] Keep HTTP and Crimson Mac/VPN validation as a separate consumer-side
      evidence tier.
- [x] Derive chunks and shards from uncompressed bytes, dtype, record shape,
      access pattern, and lifecycle; never pass benchmark row-count constants.
- [x] Keep logical dtype fixed while comparing physical storage candidates.
- [x] Use Palette's shared `fisheye.cluster.lsf` kernel rather than adding a new
      direct `bsub` implementation or job-ID parser.
- [x] Run competing candidates within the same allocation when practical so
      host and cluster-load differences do not become layout differences.
- [x] Use a fresh subprocess per candidate so peak RSS and process-local cache
      state remain attributable.
- [x] Consolidate and validate metadata only from one authoritative publication
      process after payload completion.

## Non-Goals

- Rewriting the historical source run in place.
- Benchmarking against incoming or actively produced analysis data.
- Promoting any benchmark candidate to canonical, training, or selector
  authority.
- Comparing `float16`, quantized geometry, or other logical representations.
- Treating node-local or PRFS results as proof of HTTP/Mac performance.
- Migrating production writers before the benchmark promotion gate passes.

## Required Artifact Layout

The LSF workflow must produce one durable submission bundle:

```text
<run-dir>/
├── plan.json
├── submission.json
├── matrix.json
├── cases.jsonl
├── reports/
├── status/
├── logs/
└── summary.json
```

Published benchmark candidates must use a separate shared namespace:

```text
<benchmark-root>/canonical_detection_storage/<workflow-id>/
└── <case-id>.zarr
```

Node-local scratch contains staged sources, canonical local inputs, and
ephemeral local candidates. It is removed after runtime evidence is captured
unless the operator explicitly requests retention.

## Phase 0 — Preserve And Refactor The Single-Case Foundation

- [x] Resolve exact Zarr v3 data and shard-index codec chains.
- [x] Create arrays through the policy-owned array factory.
- [x] Convert the historical detection representation to the exact canonical
      nine-array schema.
- [x] Validate every destination array by exact digest.
- [x] Require fresh destinations below a benchmark-only root.
- [x] Record initial 200,000-frame regular-versus-sharded smoke evidence.
- [x] Separate reusable source preparation, candidate execution, workload
      execution, reporting, and CLI orchestration from the initial diagnostic.
- [x] Preserve the existing single-candidate CLI as a thin adapter.
- [x] Add JSON validation for the common benchmark envelope.

Exit gate:

- [x] A matrix runner can invoke one candidate without importing CLI parsing or
      duplicating storage, schema, codec, or digest logic.

## Phase 1 — Cluster-Visible Safe Fixture

- [ ] Create a cluster-visible safe fixture below the approved shared benchmark
      namespace.
- [ ] Copy from the selected historical source in read-only mode.
- [ ] Mark the copy noncanonical, unregistered, selector-ineligible, and
      benchmark-only.
- [ ] Record source and copied paths, Zarr version, schema observations, file
      count, apparent bytes, and tree digest.
- [ ] Verify exact relative-path and content equality after copying.
- [ ] Make the fixture immutable to benchmark jobs.
- [ ] Confirm no benchmark command can resolve a canonical recording path as a
      writable destination.
- [ ] Record a stable fixture ID used by every case manifest.

Exit gate:

- [ ] The fixture can be staged by an LSF job without reading any mutable or
      actively produced dataset.

## Phase 2 — Matrix Planning And Deduplication

- [x] Add versioned matrix, scale, candidate, repetition, and workload models.
- [x] Define initial frame scales: `200,000` and the full representative
      `1,188,000` frames.
- [x] Define inner-chunk targets: `128 KiB`, `512 KiB`, `1 MiB`, and `2 MiB`.
- [x] Define regular and indexed-sharded layouts.
- [x] Define shard targets: `8 MiB`, `32 MiB`, `128 MiB`, and `512 MiB` where
      they produce distinct physical plans.
- [x] Resolve every candidate through the shared storage planner.
- [x] Fingerprint the complete stage plan, codec profile, logical schema, and
      scale.
- [x] Deduplicate labels that resolve to identical physical plans.
- [x] Explain every removed duplicate in `matrix.json`.
- [x] Reject raw per-array chunk or shard row overrides.
- [x] Emit exact expected destinations and collision results in plan-only mode.
- [x] Generate a deterministic balanced trial order from a recorded seed.
- [x] Ensure regular and sharded layouts appear in early and late positions
      across repetitions.
- [x] Record correctness gates and performance tolerances before executing the
      measured matrix.

Exit gate:

- [x] The planner produces a stable, JSON-safe set of unique physical cases
      without reading or writing Zarr payloads.

## Phase 3 — Stage And Local Canonical Preparation

- [ ] Allocate a unique scratch root from `/scratch/$USER/$LSB_JOBID` or the
      LSF-provided local temporary directory.
- [ ] Refuse shared-storage paths as scratch roots.
- [ ] Copy the safe fixture to scratch once per scale/repetition allocation.
- [ ] Record stage-in elapsed time, bytes, file count, tool, and return code.
- [ ] Validate the staged tree against the fixture manifest before computation.
- [ ] Open the staged source with direct metadata in read-only mode.
- [ ] Convert once to the exact canonical schema and dtype set.
- [ ] Materialize one fixed local canonical staging store.
- [ ] Record all legacy-to-canonical conversions in provenance.
- [ ] Validate canonical schema, row ordering, offsets, geometry, dtypes, and
      array digests before starting candidate timers.
- [ ] Ensure every candidate reads this same local canonical staging store.
- [ ] Exclude stage-in and canonical preparation from candidate write and
      publication summaries while reporting them separately.

Exit gate:

- [ ] Candidate execution cannot reach the shared source fixture after local
      preparation succeeds.

## Phase 4 — Local Materialization Matrix

- [ ] Create every local candidate at a new scratch path.
- [ ] Execute each candidate in a separate subprocess.
- [ ] Fix native thread counts and record their effective values.
- [ ] Write regular arrays by complete physical chunks.
- [ ] Write sharded arrays by complete, non-overlapping outer shards.
- [ ] Record requested and effective write ownership.
- [ ] Measure array creation, payload write, total materialization, logical
      throughput, physical throughput, peak RSS, object count, and compression.
- [ ] Record actual metadata codecs and chunk grids from the written Zarr.
- [ ] Validate exact schema, dtype, shape, and value digests.
- [ ] Fail the case if any required payload is missing or any codec differs from
      its versioned profile.
- [ ] Retain local candidates until their local-read workloads and publication
      inputs are complete.

Exit gate:

- [ ] Every successful local result is exact and comparable through the common
      benchmark envelope.

## Phase 5 — Publish Back To Shared Storage

- [ ] Give every candidate a fresh destination below the approved shared
      benchmark workflow root.
- [ ] Refuse destinations inside recording analysis Zarrs, training roots,
      selector targets, or the source fixture.
- [ ] Write initially to a case-specific incomplete destination.
- [ ] Benchmark publication from the fixed local canonical staging store into
      the candidate's final immutable plan.
- [ ] Measure source inventory, decode/re-encode, transfer, validation,
      consolidation, completion, and total publication time separately.
- [ ] Record logical bytes, transferred bytes where observable, created files,
      allocated bytes, and peak RSS.
- [ ] Use one publisher initially.
- [ ] For shortlisted candidates, benchmark one, two, and four workers with
      whole-chunk or whole-shard ownership only.
- [ ] Validate the published destination directly before metadata
      consolidation.
- [ ] Consolidate metadata from one authoritative process.
- [ ] Reopen through consolidated metadata and compare expected schema, dtype,
      shapes, chunk grids, shards, and codecs with actual metadata.
- [ ] Validate exact array digests from the published store.
- [ ] Write a completion record only after all validations pass.
- [ ] Finalize the benchmark destination without updating registry or selector
      state.
- [ ] Preserve failed destinations as visibly incomplete evidence or remove
      them only through an explicit cleanup command.

Exit gate:

- [ ] A successful publication is independently readable, exact, complete, and
      impossible to confuse with canonical data.

## Phase 6 — Read Workloads

Local and published reads are different benchmark tiers and must remain
separate in reports.

- [ ] Run each read workload in a fresh subprocess for process-cold timing.
- [ ] Repeat within the process for warm timing.
- [ ] Do not label OS or shared-filesystem cache state cold unless cache control
      is both real and recorded.
- [ ] Measure direct and consolidated metadata opening separately.
- [ ] Read the complete eager `frame_row_offsets` index.
- [ ] Read two adjacent offsets followed by the selected per-frame instance
      slice.
- [ ] Run deterministic random-frame reads.
- [ ] Run sequential frame windows and 700-FPS traversal.
- [ ] Run contiguous and random observation-row reads used by joins.
- [ ] Run complete-array scans.
- [ ] Use identical frame, row, window, order, and seed inputs for every
      candidate.
- [ ] Record latency distributions, logical bytes returned, decoded bytes where
      observable, request/range count where observable, and throughput.
- [ ] Run local-scratch workloads before cleanup.
- [ ] Run PRFS workloads only against successfully published benchmark outputs.

Exit gate:

- [ ] Every read result identifies its storage tier and cache condition without
      using local evidence as a remote-performance claim.

## Phase 7 — LSF Workflow

- [x] Add a concurrency-safe helper that deploys a clean agent branch as a
      detached, commit-pinned `/groups` worktree without switching the shared
      checkout.
- [ ] Implement a benchmark-family planner on `fisheye.cluster.lsf` models,
      bundle persistence, submission, task groups, and runtime status.
- [ ] Keep the operator shell surface thin and free of new `bsub` parsing.
- [ ] Make plan-only mode the default and require explicit `--submit`.
- [ ] Record the cluster-visible Palette repository, exact commit, branch, and
      clean-state check.
- [ ] Refuse execution when the job commit differs from the planned commit.
- [ ] Use a CPU allocation; request no GPU.
- [ ] Represent one `(scale, repetition)` block as one LSF array element.
- [ ] Run all unique candidates for that block on the same execution host.
- [ ] Execute candidates sequentially in the planned balanced order while
      retaining subprocess isolation.
- [ ] Bound active LSF array elements explicitly.
- [ ] Record job ID, array index, job name, queue, host, slots, memory request,
      walltime, scratch root, command, timestamps, and return code.
- [ ] Write atomic running, succeeded, or failed status for every array element.
- [ ] Install signal and failure traps that retain status and logs.
- [ ] Clean node-local scratch on exit by default without deleting published
      benchmark evidence.
- [ ] Submit one success-gated serial finalizer using structured `done`
      dependencies.
- [ ] Have the finalizer validate expected reports and aggregate results only;
      it must not update registry or selectors.

Exit gate:

- [ ] Fake-runner tests prove plan, submission, array selection, failure status,
      dependency, and finalizer behavior before a real `bsub` smoke.

## Phase 8 — Cluster Smoke And Bounded Rollout

- [x] Render and review one complete plan with no submission.
- [x] Run one small local end-to-end fixture outside the sandbox.
- [ ] Submit one `200,000`-frame scale/repetition block.
- [ ] Verify stage-in, scratch use, candidate order, reports, publish paths,
      validation, cleanup, and finalizer evidence.
- [ ] Confirm the source fixture remains unchanged.
- [ ] Confirm no analysis archive, registry row, selector, or training artifact
      changed.
- [ ] Measure actual resource use and revise LSF memory/walltime requests.
- [ ] Submit the remaining bounded `200,000`-frame repetitions.
- [ ] Summarize variability before expanding to full duration.

Exit gate:

- [ ] The limited cluster run is exact, operationally diagnosable, and safely
      isolated from canonical data.

## Phase 9 — Full Matrix And Candidate Reduction

- [ ] Complete at least five balanced repetitions for every unique 200,000-frame
      physical plan.
- [ ] Reject incorrect, dominated, or operationally unsafe candidates.
- [ ] Record rejected candidates and reasons; do not silently omit them.
- [ ] Carry only the Pareto frontier and required controls to full duration.
- [ ] Run full-duration materialization, publication, and read workloads.
- [ ] Benchmark parallel publication only for finalists.
- [ ] Report median, p95 where meaningful, dispersion, trial order, and host
      identity rather than only the fastest observation.
- [ ] Compare object count, local write, PRFS publication, PRFS reads, and peak
      memory as separate dimensions.
- [ ] Select the fewest-object candidate that satisfies all predeclared
      correctness, latency, throughput, and resource gates.

Exit gate:

- [ ] No cluster-only result is promoted until consumer-side validation is
      complete.

## Phase 10 — HTTP And Crimson Validation

- [ ] Serve finalist stores through a request-logging HTTP Range path.
- [ ] Record metadata requests, range requests, transferred bytes, decoded
      bytes, latency, and read amplification.
- [ ] Verify the exact Zarr v3 codec and shard-index chain in the Crimson driver.
- [ ] Verify direct and consolidated metadata behavior explicitly.
- [ ] Measure Crimson initialization, random scrub, forward playback, frame
      windows, and full-array operations.
- [ ] Run the actual Mac/VPN path used in practice.
- [ ] Record unsupported consolidated-metadata behavior as a compatibility gate,
      not as a Python-only success.
- [ ] Compare results against the predeclared acceptance thresholds.

Exit gate:

- [ ] Palette and Crimson both validate the selected candidate and its workload
      performance on the intended delivery path.

## Phase 11 — Profile Promotion And Reuse

- [ ] Give the selected immutable profile a new versioned identity.
- [ ] Preserve the complete matrix, environment, source, rejection, and
      selection evidence.
- [ ] Update production integration only after profile review.
- [ ] Keep editable authorities unsharded unless their editing contract changes.
- [ ] Use the same stage-local-compute-publish workflow when promoting editable
      keypoints or masks into immutable sharded training Zarrs.
- [ ] Add one dataset-family adapter at a time for keypoints, masks, contours,
      timelines, and other inventoried arrays.
- [ ] Require every adapter to declare its logical schema and real consumer
      workload while reusing the shared matrix and LSF machinery.

Final exit gate:

- [ ] The chosen detection profile is reproducible, versioned, cluster-tested,
      Crimson-tested, and ready for production-writer adoption without making
      benchmark data authoritative.
