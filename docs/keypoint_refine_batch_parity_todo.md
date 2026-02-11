# Keypoint Refine Batch Parity TODO

Purpose: align keypoint refinement batch operations with the detect refinement
batch operator workflow so large reruns are predictable, auditable, and easy
to drive from CLI.

Design constraints:
- migration-safe (no schema churn required for first pass)
- operator-first (dry-run, explicit scope filters, deterministic logs)
- preserve current refine algorithm semantics (this is orchestration parity)

## Why This Exists

Today:
- Detect has a dedicated refine batch entrypoint:
  - `fisheye.utils.refine_detect_batch`
- Keypoints uses `fisheye.utils.run_keypoints_batch` for both detection and
  refinement (`--refine`, `--refine-only`).

This works, but refinement-only operations are not contract-parity with detect
batch and are harder to reason about for bulk reruns.

## Current Gaps (Detect vs Keypoint Refine Batch)

- [x] No dedicated `refine_keypoints_batch` command.
- [x] No `--zarr-use analysis|training|any` filter in keypoint refine-only flow.
- [x] No explicit batch-level `--keypoint-run` selector contract.
- [x] Discovery mismatch risk:
  - root/H5 discovery path derives `<h5_stem>.zarr`,
  - archives are often `_analysis.zarr` / `_training.zarr`.
- [x] Process model mismatch:
  - detect refine batch executes per-archive subprocesses,
  - keypoint refine-only currently runs in-process.

## Target Contract

New command:
- [x] `scripts/py -m fisheye.utils.refine_keypoints_batch`

Expected interface parity with detect refine batch:
- [x] `paths [..]`
- [x] `--recursive`
- [x] `--apply` (default dry-run)
- [x] `--zarr-use analysis|training|any` (default: `analysis`)
- [x] `--no-skip-existing`
- [x] `--log-dir`

Keypoint-specific refinement control:
- [x] `--keypoint-run <name>` (default: latest in `keypoints_runs`)
- [x] optional passthrough for refine options (if exposed in
      `fisheye.refinement.refine_keypoints` CLI)

## Rollout Plan

### Phase 1: Dedicated Refine-Only Batch Tool

- [x] Add `src/fisheye/utils/refine_keypoints_batch.py`.
- [x] Mirror detect-batch dry-run plan output and summary counters:
  - `ok`, `skipped`, `missing`, `failed`.
- [x] Add JSONL logging style parity with detect batch.

### Phase 2: Zarr-First Discovery + Scope Filters

- [x] Use zarr-first discovery (`*/zarr/*.zarr`, recursive variant).
- [x] Add `--zarr-use` filtering based on purpose/name inference.
- [x] Ensure explicit zarr paths and file-list paths are supported.

### Phase 3: Run Selection and Existing-Run Behavior

- [x] Implement `--keypoint-run` selection.
- [x] Skip behavior parity:
  - default skip if refined keypoints already present,
  - `--no-skip-existing` to force rerun.

### Phase 4: Execution Model Parity

- [x] Run each refinement in a subprocess:
  - isolate failures,
  - avoid long-lived process state accumulation.
- [x] Ensure command echo + return-code capture parity with detect batch.

### Phase 5: Integration Cleanup

- [ ] Optionally make `run_keypoints_batch --refine-only` delegate to the new
      `refine_keypoints_batch` implementation.
- [ ] Keep `run_keypoints_batch --refine` for convenience orchestration.
- [x] Mark `run_keypoints_batch --refine-only` as deprecated (warn users and
      point to `refine_keypoints_batch`).

## Testing Plan

- [x] Add `tests/unit/fisheye/test_refine_keypoints_batch.py`.
- [ ] Cover:
  - plan building for missing/skipped/ok,
  - `--zarr-use` filtering,
  - explicit `--keypoint-run` resolution,
  - dry-run determinism,
  - apply path subprocess invocation and failure accounting.

## Acceptance Criteria

- [ ] Operators can rerun keypoint refinement in bulk without relying on
      H5-derived zarr naming.
- [ ] CLI scope and lifecycle controls match detect refine batch expectations.
- [ ] Repeated dry-runs produce stable counts and planned targets.
- [ ] Apply mode is auditable via JSONL logs and per-recording return status.
- [ ] Existing `run_keypoints_batch` user workflows remain functional.

## Open Decisions

- [x] `run_keypoints_batch --refine-only` should be marked deprecated after
      delegation.
- [x] Default scope should remain `analysis` only.
- [x] Keep detect-style aggregate failure summary as the default behavior
      (no strict fail-fast default).
