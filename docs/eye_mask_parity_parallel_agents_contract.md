# Eye-Mask Parity Parallel Agent Contract

Purpose: define a conflict-safe parallel execution contract for implementing
`docs/eye_mask_parity_todo.md` with deterministic delivery gates.

## Scope

In scope:
- implement P1-P6 work from `docs/eye_mask_parity_todo.md`.
- define strict parallel ownership by module/CLI surface.
- enforce compatibility contracts between registry profile, data card, plotting,
  pipeline orchestration, maintenance, and query/check UX.
- require targeted test evidence per surface plus end-to-end acceptance evidence.

Out of scope:
- revisiting completed historical detect/pose parity tasks in
  `docs/eye_masks_detect_pose_parity_todo.md`.
- unrelated registry normalization or non-eye-mask training workflows.
- dependency installation or environment mutation outside approved workflows.

## Source Of Truth

- Backlog: `docs/eye_mask_parity_todo.md`
- Historical context: `docs/eye_masks_detect_pose_parity_todo.md`
- Registry storage baseline: `src/fisheye/registry/db.py`
- Maintenance baseline: `src/fisheye/registry/maintenance.py`
- Query/check surfaces:
  - `src/fisheye/utils/registry_query.py`
  - `src/fisheye/utils/check_training_registry.py`
- Eye-mask export/pipeline surfaces:
  - `src/fisheye/utils/export_eye_mask_training_zarr.py`
  - `src/fisheye/core/pipeline.py`

## Canonical Task IDs

- `EM-A`: profile registry contract + storage + sync command.
- `EM-B`: eye-mask data-card contract + aggregation + plotting.
- `EM-C`: export/pipeline orchestration + merged-export wiring.
- `EM-D`: maintenance backfill/refresh command parity + operator runbook.
- `EM-E`: query/check UX parity for eye-mask profile/performance surfaces.
- `EM-F`: acceptance integration, validation evidence, TODO closeout.

## Agent Ownership (Strict)

No cross-task edits outside owned files without explicit handoff.
Only `EM-F` updates status checkboxes in `docs/eye_mask_parity_todo.md`.

### Agent A (`EM-A`: Registry Profile Contract + Storage + Sync)

Owns:
- new profile contract doc under `docs/` (recommended:
  `docs/eye_mask_data_profile_schema_contract.md`).
- `src/fisheye/registry/db.py` (eye-mask profile table/view/index additions).
- new sync CLI module (recommended:
  `src/fisheye/utils/sync_eye_mask_profile_registry.py`).
- profile-registry tests under `tests/unit/fisheye/` (new files allowed).

Must deliver:
- canonical eye-mask profile schema contract and required run attributes.
- registry table + latest views + indexes for eye-mask profiles.
- deterministic upsert/replace behavior for profile rows.
- sync command with dry-run/apply modes and clear summary counts.
- fail-closed freshness metadata fields required by downstream consumers.

### Agent B (`EM-B`: Data Card + Plot Bundle)

Owns:
- new data-card contract doc under `docs/` (recommended:
  `docs/eye_mask_training_data_card_contract.md`).
- new aggregation CLI module (recommended:
  `src/fisheye/utils/aggregate_eye_mask_training_data_card.py`).
- new plotting CLI module (recommended:
  `src/fisheye/utils/plot_eye_mask_training_data_card.py`).
- data-card/plot tests under `tests/unit/fisheye/` (new files allowed).

Must deliver:
- schema-compliant eye-mask training data-card payload.
- registry-first aggregation read path with explicit fallback policy.
- deterministic plot bundle output paths and filenames.
- parity sections required in TODO P2 (`selection`, `quality`, `geometry`,
  `spatial`, `composition`, `subject_coverage`, `parity`, `audit_freshness`).

### Agent C (`EM-C`: Pipeline + Merged Export Orchestration)

Owns:
- `src/fisheye/utils/export_eye_mask_training_zarr.py`
- `src/fisheye/core/pipeline.py`
- pipeline/export tests in `tests/unit/fisheye/` for these modules.

Must deliver:
- merged-export registry registration parity for eye-mask outputs.
- pipeline flags aligned with detect/keypoint ergonomics for aggregation
  enable/disable behavior.
- pre-aggregation gating that triggers profile sync or fails with remediation.
- deterministic metadata linkage from export outputs to card/plot artifacts.

### Agent D (`EM-D`: Maintenance + Backfill/Refresh + Ops Runbook)

Owns:
- `src/fisheye/registry/maintenance.py` (eye-mask profile paths only).
- optional operator runbook doc under `docs/` for command procedures/results.
- maintenance tests in `tests/unit/fisheye/test_registry_maintenance.py`
  (eye-mask profile focused).

Must deliver:
- new maintenance commands for eye-mask profile backfill/refresh parity.
- deterministic dry-run/apply summaries (`inserted/updated/deleted/unchanged`).
- idempotent rerun behavior and stale-row reconciliation behavior.
- operator runbook with one-time backfill and routine refresh procedures.

### Agent E (`EM-E`: Query/Check UX Parity)

Owns:
- `src/fisheye/utils/check_training_registry.py`
- `src/fisheye/utils/registry_query.py`
- tests:
  - `tests/unit/fisheye/test_check_training_registry.py`
  - `tests/unit/fisheye/test_registry_query.py`

Must deliver:
- dedicated eye-mask `check_training_registry --view` surfaces for
  profile/performance parity.
- detail-table parity for staleness, exclusion reasons, and review rollups.
- query output parity for profile-linked filters and machine-readable exports.
- actionable remediation messages for missing/stale profile rows.

### Agent F (`EM-F`: Acceptance Integration + Closeout)

Owns:
- `docs/eye_mask_parity_todo.md` checkbox/state updates.
- optional top-level acceptance evidence doc under `docs/`.
- cross-surface acceptance tests/harness docs (without reworking owned modules).

Must deliver:
- integration validation for one representative eye-mask training set:
  merged export -> profile sync -> data card -> plots -> query/check review.
- acceptance evidence bundle with commands, outputs, and pass/fail summary.
- final TODO updates with evidence-linked checkboxes only after validations pass.

## Shared Interface Freeze (Required Before Parallel Coding)

`EM-A`, `EM-B`, and `EM-C` must agree on these interfaces before broad coding:
- profile table/view names and required columns.
- profile freshness/staleness semantics and expected remediation command text.
- data-card schema name/version and required section keys.
- artifact path conventions for card JSON and plot outputs.
- canonical command names:
  - `scripts/py -m fisheye.utils.sync_eye_mask_profile_registry`
  - `scripts/py -m fisheye.utils.aggregate_eye_mask_training_data_card`
  - `scripts/py -m fisheye.utils.plot_eye_mask_training_data_card`

No agent may rename these after integration starts without explicit cross-agent
handoff and contract update.

## Parallel Execution Plan

Wave 0 (contract freeze):
- `EM-A` publishes schema/storage contract draft.
- `EM-B` publishes data-card schema draft aligned to profile fields.
- `EM-C` publishes pipeline flag and orchestration draft.
- Gate: interface freeze approved.

Wave 1 (independent implementation):
- `EM-A` implements registry profile table/views + sync CLI.
- `EM-B` implements aggregation + plotting modules.
- `EM-D` implements maintenance flags + idempotent behavior.
- `EM-E` implements query/check UX surfaces using agreed interfaces.

Wave 2 (integration):
- `EM-C` integrates export/pipeline orchestration against completed commands.
- `EM-F` runs end-to-end acceptance and compiles evidence.

Wave 3 (closeout):
- `EM-F` updates `docs/eye_mask_parity_todo.md` status checkboxes with evidence.

## Per-Agent Process Contract

Each agent follows this sequence:
1. Confirm owned files only.
2. Implement minimum behavior needed for assigned parity gates.
3. Add or update targeted tests.
4. Run targeted commands with `scripts/py -m pytest ...`.
5. Provide handoff note:
   - task id
   - files touched
   - behavior changes
   - commands run
   - summary counts/results
   - residual risks.

## Integration Contract

Merge order:
1. `EM-A`, `EM-B`, `EM-D`, and `EM-E` in parallel after interface freeze.
2. `EM-C` rebases on merged outputs from `EM-A` + `EM-B` + `EM-D`.
3. `EM-F` validates and closes TODO checkboxes.

Conflict policy:
- no opportunistic refactors outside owned surfaces.
- if a non-owned edit is required, pause and request explicit handoff.
- if contract assumptions diverge, stop and update this contract first.

## Validation Gates

Required:
- targeted unit tests pass for each owned surface.
- profile sync and maintenance commands show deterministic rerun behavior.
- query/check surfaces correctly report missing/stale profile remediation.
- end-to-end workflow succeeds for one representative training set.

Recommended command set (examples):

```bash
scripts/py -m pytest tests/unit/fisheye/test_registry_maintenance.py -k eye_mask
scripts/py -m pytest tests/unit/fisheye/test_check_training_registry.py
scripts/py -m pytest tests/unit/fisheye/test_registry_query.py
scripts/py -m pytest tests/unit/fisheye/test_core_pipeline_refined_eye_masks_stage.py
scripts/py -m pytest tests/unit/fisheye/test_validate_eye_mask_training_zarr.py
```

Sandbox policy for zarr-heavy tests:
- follow repository sandbox zarr guidance.
- prefer in-memory/fake-group harnesses for unit coverage.
- if a real-zarr test path hangs in sandbox, defer that path and record exact
  local validation commands in handoff evidence.

## Acceptance Exit Criteria

All of the following must be true before closing the TODO:
- P1-P6 items in `docs/eye_mask_parity_todo.md` are validated and checked.
- command surfaces are discoverable and aligned with detect/keypoint ergonomics.
- registry/profile/card/plot/pipeline/maintenance/query/check paths are
  end-to-end executable without bespoke one-off scripts.
