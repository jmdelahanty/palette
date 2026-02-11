# Crimson Detect Review Acceptance Contract

Purpose: define a migration-safe, operator-first contract for approving detect
review status from Crimson after inspection/manual edits.

Date anchored: 2026-02-09.

## Scope

- In scope:
  - Writing review acceptance metadata on refined detect runs
  - Deterministic target-run/group resolution
  - Wrapper behavior for Crimson integration
- Out of scope:
  - Editing detection arrays (handled by manual write contract)
  - Running detect/refine stages
  - Registry schema changes

Related docs:
- `docs/crimson_refined_detect_manual_contract.md`
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
2. Resolve target detect group using preference chain:
   - `manual -> interpolated -> filtered -> raw`
   - or explicit `--target-group`.
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
- `resolved_group`: resolved group label
- `preference_chain`: current detect-group preference chain

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
- `--target-group <manual|interpolated|filtered|raw|custom>` (optional)
- `--state <approved|pending|rejected|needs_review>` (default: approved)
- `--method <manual|algorithmic|hybrid|spotcheck>` (default: manual)
- `--intended-use <training|full_recording>` (required in strict mode)
- `--reviewer <id>` (recommended; required for approve in strict mode)
- `--notes <text>`
- `--strict` (enable policy guardrails)
- `--no-latest` (do not move parent latest pointer)
- `--dry-run` (show resolved target + payload only)
- `--json` (machine-readable output)

Output contract:
- Human-readable summary by default.
- JSON payload with:
  - `zarr_path`, `refined_run`, `resolved_group`, `state`, `method`,
    `intended_use`, `reviewer`, `latest_updated`, `dry_run`.

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
2. `resolved_group` equals expected group.
3. Parent latest pointer updated unless `--no-latest`.
4. `check_recording_steps` shows updated detect review state.

## Non-Goals / Stability Notes

- This contract does not alter detect/refine data models.
- This contract does not require DB migration.
- Field names must remain aligned with existing Palette readers; do not invent
  alternate keys.
