# Crimson Detect Review Acceptance Contract
<!-- contract-meta
version: 2
status: active
last_verified: 2026-04-15
-->

Purpose: define a migration-safe, operator-first contract for approving detect
review status from Crimson after inspection/manual edits.

Date anchored: 2026-04-11.

## Scope

- In scope:
  - Writing review acceptance metadata on refined detect runs
  - Deterministic target-run/current-surface resolution
  - Wrapper behavior for Crimson integration
- Out of scope:
  - Editing detection arrays (handled by manual write contract)
  - Running detect/refine stages
  - Registry schema changes

Related docs:
- `docs/archive/crimson_refined_detect_manual_contract.md`
- `docs/crimson_detect_bbox_read_contract.md`
- `docs/detection_refinement_workflow.md`

## Existing Primitive

Palette already provides:
- `fisheye.utils.set_detect_review_status`

This contract treats that module as the source of truth for status payload
shape and attr write locations.

## Acceptance Model

Acceptance is metadata-only and append-safe:
1. Select refined run (`latest` unless explicitly specified).
2. Resolve the target detect surface:
   - current runs should resolve to `refined`, meaning the canonical curated
     sparse surface on `refined_detect_runs/<run>/instances`
   - legacy subgroup fallback (`manual -> interpolated -> filtered -> raw`)
     should only be used for historical archives or explicit compatibility
     reads
   - explicit `--target-group` may override this when the operator is working
     with a historical sparse subgroup
3. Write `detect_review_status` onto selected refined run.
4. Update parent pointer:
   - `refined_detect_runs.attrs["detect_review_status_latest"] = <run_name>`

No raw detect arrays are mutated.

## Required Payload

Required fields:
- `state`: one of `approved | pending | rejected | needs_review`
- `method`: one of `manual | algorithmic | hybrid | spotcheck`
- `intended_use`: one of `training | full_recording`
- `timestamp`: ISO UTC timestamp
- `resolved_group`: resolved logical detect surface label
  - should normally be `refined` for current runs
- `preference_chain`: current detect-group preference chain
  - current default should begin with `refined`
  - subgroup-era entries are compatibility fallbacks, not the primary current
    mode

Recommended fields:
- `reviewer`: operator identity
- `notes`: short rationale/context
- `target_group`: explicit override group when used

## Guardrails (Fail-Closed)

Wrapper MUST fail with non-zero and clear error when:
1. No refined run exists.
2. Explicit refined run is missing.
3. Explicit target group is requested but not resolvable.
4. `state=approved` but resolved group is empty.
5. Reviewer is required by policy and not provided.

Recommended policy defaults:
- Require `--reviewer` for `approved`.
- Require `--notes` for `rejected`.

## CLI Wrapper Plan

Implement a thin wrapper for Crimson/operator use:
- Module: `fisheye.utils.accept_detect_review`
- Behavior: validate policy + delegate to `set_detect_review_status` semantics.

Proposed arguments:
- `zarr_path`
- `--refined-run <name>` (optional)
- `--target-group <refined|manual|interpolated|filtered|raw|custom>` (optional)
  - `refined` is the normal current value
  - subgroup values are for historical sparse compatibility only
- `--state <approved|pending|rejected|needs_review>` (default: approved)
- `--method <manual|algorithmic|hybrid|spotcheck>` (default: manual)
- `--intended-use <training|full_recording>` (required in strict mode)
- `--reviewer <id>` (recommended; required for approve in strict mode)
- `--notes <text>`
- `--strict` (enable policy guardrails)
- `--dry-run` (show resolved target + payload only)
- `--json` (machine-readable output)

Output contract:
- Human-readable summary by default.
- JSON payload with:
  - `zarr_path`, `refined_run`, `resolved_group`, `state`, `method`,
    `intended_use`, `reviewer`, `authoritative_approval`, `dry_run`.

## Crimson Integration Pattern

Recommended flow from Crimson:
1. Operator inspects detect overlays.
2. Operator performs manual edits (if needed) per manual-write contract.
3. Crimson invokes wrapper with explicit intent:
   - approved training example:
     - `... --state approved --method manual --intended-use training --reviewer <id>`
   - approved full-recording example:
     - `... --state approved --method manual --intended-use full_recording --reviewer <id>`
4. Crimson stores command + response in its own action log.

## Validation Checklist

After acceptance write:
1. `refined_detect_runs/<run>.attrs["detect_review_status"]` exists.
2. `resolved_group` equals the expected surface, normally `refined`.
3. For `--state approved`: `refined_detect_runs.attrs["authoritative_run"]` points at
   the run (approval routes through `palette approve`; the legacy
   `detect_review_status_latest` pointer is no longer written — 2026-07-05).
4. `check_recording_steps` shows updated detect review state.

## Non-Goals / Stability Notes

- This contract does not alter detect/refine data models.
- This contract does not require DB migration.
- Field names must remain aligned with existing Palette readers; do not invent
  alternate keys.
