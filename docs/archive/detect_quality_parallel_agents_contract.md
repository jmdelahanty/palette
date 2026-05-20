# Detect Quality Parallel Agent Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
stage_arrays_spec: DETECT_QUALITY_SPEC
-->

Purpose: define a conflict-safe parallel execution plan for the remaining
detect-quality registry TODO work after schema + write-path implementation.

## Scope

In scope:
- close remaining `docs/detect_quality_registry_todo.md` items.
- add deterministic test coverage for latest-row selection and maintenance behavior.
- capture one-time operator backfill/refresh evidence.
- add observability/reporting coverage for exclusion reasons.

Out of scope:
- unrelated recording-registry normalization tasks.
- keypoint/eye-mask parity feature work (except shared test helpers if required).

## Source Of Truth

- TODO tracker: `docs/detect_quality_registry_todo.md`
- registry schema + views: `src/fisheye/registry/db.py`
- maintenance actions: `src/fisheye/registry/maintenance.py`
- detect training preflight:
  - `src/fisheye/utils/prepare_detect_training_from_registry.py`
  - `src/fisheye/utils/run_detect_training_pipeline.py`
- operator/query surfaces:
  - `src/fisheye/utils/registry_query.py`
  - `src/fisheye/utils/check_training_registry.py`

## Canonical Task IDs

- `DQ-A`: deterministic schema/view/extraction tests.
- `DQ-B`: maintenance determinism + backfill/refresh evidence.
- `DQ-C`: fail-closed preflight parity + exclusion observability.

## Agent Ownership (Strict)

No cross-task edits outside owned files without explicit handoff.

### Agent A (`DQ-A`: Schema/View Determinism)

Owns:
- `src/fisheye/registry/db.py` (only detect-quality view/extraction related edits).
- `tests/unit/fisheye/test_registry_detect_performance.py`
- `tests/unit/fisheye/test_registry_query.py`

Must deliver:
- deterministic `refined_detect_review_current` selection when multiple refined runs exist
  (`detect_quality_current` remains the compatibility alias).
- tie-break tests for:
  - `review_timestamp_utc`
  - fallback to `refined_created_utc`
  - fallback to `refined_run` lexical order when timestamps tie/missing.
- extraction tests with multiple refined runs and mixed review payloads.

### Agent B (`DQ-B`: Maintenance Determinism + Ops Evidence)

Owns:
- `src/fisheye/registry/maintenance.py` (detect-quality path only).
- `tests/unit/fisheye/test_registry_maintenance.py`
- optional runbook note under `docs/` for operator commands/results.

Must deliver:
- repeatable dry-run/apply counts for detect-quality backfill/refresh.
- tests proving idempotent behavior across repeated runs.
- tests for stale-row deletion behavior in refresh mode.
- operator evidence template:
  - command(s)
  - inserted/updated/deleted/unchanged counts
  - follow-up verification query output.

### Agent C (`DQ-C`: Preflight + Observability)

Owns:
- `src/fisheye/utils/prepare_detect_training_from_registry.py`
- `src/fisheye/utils/run_detect_training_pipeline.py`
- `src/fisheye/utils/check_training_registry.py`
- related tests under `tests/unit/fisheye/` for those modules.

Must deliver:
- explicit fail-closed checks that selected registry rows still match current Zarr metadata.
- clear exclusion reason reporting:
  - missing review
  - wrong state/use
  - interpolation threshold failure
  - stale/changed registry row.
- tests proving preflight rejects stale/divergent rows.

## Per-Agent Process Contract

Each agent follows this sequence:

1. Confirm owned files only.
2. Implement the minimal required behavior.
3. Add/update targeted tests.
4. Run targeted tests with `scripts/py -m pytest ...`.
5. Produce handoff note:
   - task id
   - files touched
   - behavior changes
   - commands run
   - result summary
   - remaining risks.

## Integration Contract

Merge order:
1. `DQ-A` and `DQ-B` in parallel.
2. `DQ-C` on top of merged `DQ-A` + `DQ-B` (so preflight/reporting uses final semantics).
3. update `docs/detect_quality_registry_todo.md` checkboxes with concrete evidence.

Conflict policy:
- if a non-owned file change is needed, pause and request handoff.
- avoid opportunistic refactors in unrelated modules.

## Validation Gates

Required:
- targeted unit tests for each owned module pass.
- detect-quality maintenance dry-run/apply counts are deterministic on rerun.
- preflight fail-closed checks reject intentionally stale/divergent fixtures.

Recommended commands:

```bash
scripts/py -m pytest tests/unit/fisheye/test_registry_detect_performance.py
scripts/py -m pytest tests/unit/fisheye/test_registry_maintenance.py -k detect_quality
scripts/py -m pytest tests/unit/fisheye/test_prepare_detect_training_from_registry.py
scripts/py -m pytest tests/unit/fisheye/test_run_detect_training_pipeline.py
```
