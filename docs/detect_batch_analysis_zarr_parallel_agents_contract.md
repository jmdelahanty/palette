# Detect Batch Analysis-Zarr Parallel Agent Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
stage_arrays_spec: DETECT_SPEC
-->

Purpose: define a conflict-safe parallel execution contract for shipping
registry-model-backed batch detection inference across analysis zarr archives.

## Scope

In scope:
- make batch detect planning/execution target analysis zarr archives directly.
- ensure batch execution uses registry model resolution/provenance parity.
- add cluster sharding parity for analysis-zarr batches.
- add deterministic observability and validation evidence for reruns/retries.

Out of scope:
- detect model architecture/training changes.
- keypoint/eye-mask pipeline changes outside shared status/query surfaces.
- unrelated schema migration work.

## Source Of Truth

- Batch runner: `src/fisheye/utils/run_detections_batch.py`
- Registry-backed detect runner: `src/fisheye/utils/run_detect_with_registry_model.py`
- Analysis archive contract: `docs/recording_analysis_pipeline_contract.md`
- Recording pipeline contract: `docs/recording_analysis_pipeline_contract.md`
- Cluster submission wrapper: `scripts/submit_detect_batches_bsub.sh`
- Operator guide baseline: `docs/cluster_batching_guide.md`
- Review status helper: `src/fisheye/utils/list_unapproved_analysis_zarrs.py`

## Canonical Task IDs

- `DBI-A`: analysis-zarr discovery/planning + deterministic plan semantics.
- `DBI-B`: registry-model execution parity for batch detect.
- `DBI-C`: cluster sharding/submission parity for analysis-zarr batches.
- `DBI-D`: observability, validation gates, and operator runbook closeout.

## Shared Interface Freeze (Required Before Parallel Coding)

The following interface and payload assumptions are frozen before parallel work:

- Canonical batch entrypoint remains:
  `scripts/py -m fisheye.utils.run_detections_batch`
- Batch runner emits deterministic JSONL events with at least:
  - `event`
  - `ts_utc`
  - `run_id`
  - `zarr`
  - `status`
  - `reason` (when non-success)
- Batch execution path must use registry model resolution semantics from
  `run_detect_with_registry_model` (no legacy blob-only inference path).
- Cluster wrapper must invoke repository Python via `scripts/py` (not bare
  `python`) and preserve batch run logging layout.

No agent may rename CLI surfaces or event field names after coding starts
without explicit contract update.

## Agent Ownership (Strict)

No cross-task edits outside owned files without explicit handoff.

### Agent A (`DBI-A`: Analysis-Zarr Planning)

Owns:
- `src/fisheye/utils/run_detections_batch.py` (planning/discovery/filtering only).
- `tests/unit/fisheye/test_run_detections_batch.py` (new/updated).

Must deliver:
- deterministic discovery of analysis zarr targets under provided roots.
- deterministic plan ordering and stable status reason taxonomy.
- prereq checks aligned to analysis-zarr detect execution inputs.
- dry-run/apply parity for analysis-zarr planning summaries.

### Agent B (`DBI-B`: Registry-Model Execution Parity)

Owns:
- `src/fisheye/utils/run_detect_with_registry_model.py`
- `tests/unit/fisheye/test_run_detect_with_registry_model.py`
- `tests/unit/fisheye/test_model_resolution_provenance.py`

Must deliver:
- stable callable/CLI contract suitable for batch invocation.
- explicit success/failure payload for batch caller consumption.
- deterministic provenance fields on success and clear remediation on failure.
- no silent fallback to stale/non-selected model rows.

### Agent C (`DBI-C`: Cluster Batch Submission Parity)

Owns:
- `scripts/submit_detect_batches_bsub.sh`
- `docs/cluster_batching_guide.md` (cluster usage updates only).

Must deliver:
- analysis-zarr-aware sharding input generation.
- `scripts/py`-based execution in array jobs.
- deterministic batch manifests (`recordings.txt`/`batch_*.txt`) and log paths.
- explicit dry-run output showing command, shard counts, and queue params.

### Agent D (`DBI-D`: Observability + Validation + Closeout)

Owns:
- `src/fisheye/utils/list_unapproved_analysis_zarrs.py` (if needed for gating).
- validation docs under `docs/` (new file allowed for runbook/evidence).
- test updates for observability/query behavior.

Must deliver:
- operator validation checklist for:
  - planned vs executed targets
  - success/failure/skip counts
  - rerun idempotency evidence
- clear post-run command set to identify missing/unapproved detect outputs.
- final acceptance evidence block with exact commands + summary counts.

## Parallel Execution Plan

Wave 0 (freeze):
- agree on batch JSONL event schema and execution handoff contract.

Wave 1 (parallel implementation):
- `DBI-A`, `DBI-B`, and `DBI-C` run concurrently within ownership boundaries.

Wave 2 (integration):
- rebase and wire `DBI-A` execution path to finalized `DBI-B` contract.
- apply `DBI-C` wrapper against integrated batch CLI behavior.

Wave 3 (validation/closeout):
- `DBI-D` executes validation gates and publishes operator runbook evidence.

## Per-Agent Process Contract

Each agent follows this sequence:
1. Confirm owned files only.
2. Implement minimum required behavior.
3. Add/update targeted tests.
4. Run targeted commands with `scripts/py -m pytest ...`.
5. Produce handoff note:
   - task id
   - files touched
   - behavior changes
   - commands run
   - result summary
   - residual risks.

## Integration Contract

Merge order:
1. `DBI-B` and `DBI-C` can merge independently.
2. `DBI-A` merges after rebasing on final `DBI-B` contract.
3. `DBI-D` runs validation and documentation closeout on integrated branch.

Conflict policy:
- no opportunistic refactors outside owned modules.
- if a non-owned file must change, pause and request explicit handoff.
- update this contract before changing frozen interfaces.

## Validation Gates

Required:
- planning tests verify deterministic analysis-zarr selection and status reasons.
- execution tests verify batch path uses registry-model runner semantics.
- cluster wrapper dry-run validates shard manifests and command construction.
- operator validation commands produce deterministic rerun evidence.

Recommended commands:

```bash
scripts/py -m pytest tests/unit/fisheye/test_run_detections_batch.py
scripts/py -m pytest tests/unit/fisheye/test_run_detect_with_registry_model.py
scripts/py -m pytest tests/unit/fisheye/test_model_resolution_provenance.py
scripts/py -m pytest tests/unit/fisheye/test_list_unapproved_analysis_zarrs.py
```

```bash
scripts/py -m fisheye.utils.run_detections_batch /nvme1/recordings \
  --recursive \
  --dry-run \
  --json
```

```bash
scripts/submit_detect_batches_bsub.sh \
  --root /nvme1/recordings \
  --batch-size 20 \
  --dry-run
```

## Acceptance Exit Criteria

All of the following must be true:
- batch detect targets analysis zarr archives with deterministic planning.
- execution path resolves detect models from registry with provenance parity.
- cluster wrapper and local batch CLI use the same effective execution contract.
- validation evidence includes rerun-idempotent counts and remediation commands
  for missing/unapproved outputs.
