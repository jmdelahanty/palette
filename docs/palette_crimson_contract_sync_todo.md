# Palette-Crimson Contract Sync TODO

Date anchored: 2026-03-03

## Goal

Establish palette as the single source of truth for all `palette-crimson`
contracts. The contracts repo becomes a read-only mirror of palette's
`docs/palette-crimson/` directory, synced automatically via CI. The existing
contracts repo automation (contract-trigger, question-router, clarify loop)
continues with targeted adjustments (notably answer-consumer scope + sync PR
automation) — the main difference is where `palette-crimson` edits originate.

---

## Background

### Current state

- Palette has 3 detect contracts in `docs/` with `crimson_*` naming and
  `contract-meta` blocks (version, status, last_verified, stage_arrays_spec).
- The contracts repo has copies of those 3 in `palette-crimson/` without
  `contract-meta` blocks.
- The contracts repo has 3 keypoint contracts (`keypoint_read.md`,
  `keypoint_review_acceptance.md`, `keypoint_manual_write.md`) that only exist
  there — palette has no copies.
- The contracts repo has `zarr_alignment.md` which is byte-identical to
  palette's `crimson_palette_zarr_alignment_todo.md`.
- The `palette-answer-consumer` workflow can currently edit `palette-crimson/`
  in the contracts repo, which conflicts with palette-as-author.

### Desired state

- All `palette-crimson` contracts live in `palette/docs/palette-crimson/` as
  the canonical source.
- A CI workflow in palette auto-syncs `docs/palette-crimson/` to the contracts
  repo on merge to main.
- The contracts repo's `palette-answer-consumer` is restricted from editing
  `palette-crimson/` (it only edits `citrus-palette/` and `orange-palette/`).
- The contracts repo's `contract-trigger` continues to fire when
  `palette-crimson/` changes (no change needed — it already watches that path).

---

## Phase 1: Palette-side directory setup

### 1a. Create `docs/palette-crimson/` directory

- [ ] Create `docs/palette-crimson/`
- [ ] Move and rename existing detect contracts:
  - `docs/crimson_detect_bbox_read_contract.md` →
    `docs/palette-crimson/detect_bbox_read.md`
  - `docs/crimson_detect_review_acceptance_contract.md` →
    `docs/palette-crimson/detect_review_acceptance.md`
  - `docs/crimson_refined_detect_manual_contract.md` →
    `docs/palette-crimson/refined_detect_manual.md`
- [ ] Move zarr alignment doc:
  - `docs/crimson_palette_zarr_alignment_todo.md` →
    `docs/palette-crimson/zarr_alignment.md`
- [ ] Keep `contract-meta` blocks in all moved files (they already have them).

### 1b. Import keypoint contracts from contracts repo

- [ ] Copy from contracts repo into palette:
  - `contracts/palette-crimson/keypoint_read.md` →
    `palette/docs/palette-crimson/keypoint_read.md`
  - `contracts/palette-crimson/keypoint_review_acceptance.md` →
    `palette/docs/palette-crimson/keypoint_review_acceptance.md`
  - `contracts/palette-crimson/keypoint_manual_write.md` →
    `palette/docs/palette-crimson/keypoint_manual_write.md`
- [ ] Add `contract-meta` blocks to all 3 imported files, following palette's
      convention:
  ```html
  <!-- contract-meta
  version: 1
  status: active
  last_verified: 2026-03-03
  -->
  ```
- [ ] Review imported contracts for accuracy against current palette code
      (the contracts repo versions may reference stale paths or array names).

### 1c. Update internal references

- [ ] Grep palette codebase for old filenames and update references:
  - `crimson_detect_bbox_read_contract`
  - `crimson_detect_review_acceptance_contract`
  - `crimson_refined_detect_manual_contract`
  - `crimson_palette_zarr_alignment_todo`
- [ ] Update `scripts/check_contract_freshness.py` if it hardcodes contract
      paths (check the scan directory and glob patterns).
- [ ] Update any `CLAUDE.md` or agent instruction files that reference old
      contract paths.
- [ ] Update cross-references in other docs that link to the old filenames.

---

## Phase 2: Palette CI sync workflow

### 2a. Create `palette-contract-sync.yml` in palette

- [ ] Add `.github/workflows/palette-contract-sync.yml` to palette repo.
- [ ] Trigger: push to `main` with changes in `docs/palette-crimson/**`.
- [ ] **Auth prerequisite (cross-repo PRs):**
  - Create a fine-grained PAT (or app token) that can write to
    `jmdelahanty/agent-contracts`.
  - Minimum permissions on `agent-contracts`: `Contents: Read and write`,
    `Pull requests: Read and write`, `Metadata: Read-only`.
  - Store it as a secret in the palette repo (example:
    `CONTRACTS_SYNC_TOKEN`) and use that token for create/update PR actions.
- [ ] Actions:
  1. Check out palette repo (for source files).
  2. Check out contracts repo (target).
  3. Copy `docs/palette-crimson/*` → contracts repo `palette-crimson/`.
  4. Remove any files in contracts `palette-crimson/` that no longer exist in
     palette (**full mirror with preserve list**).
  5. Preserve contracts-repo-only artifacts (do not delete/overwrite):
     `automation_smoke_test*.md`, `automation_validate_*.md`.
  6. If no changes detected, exit cleanly.
  7. Open PR to contracts repo main.
  8. Branch naming: `automation/palette-contract-sync-<sha>`.
  9. PR title: `contracts: sync palette-crimson from palette@<short-sha>`.
  10. PR body includes list of changed files and palette commit link.

### 2b. Auto-approve sync PRs in contracts repo

- [ ] Add or extend a workflow in the contracts repo to auto-approve PRs from
      the palette sync bot.
- [ ] Match criteria:
  - Branch starts with `automation/palette-contract-sync-`
  - PR actor is trusted sync identity
  - Only `palette-crimson/` files are modified
- [ ] Use token strategy that avoids self-approval failures:
  - PR creation token identity must be different from approval token identity.
  - If using `QA_PR_APPROVER_TOKEN`, ensure it is **not** the same identity as
    the PR author.
  - Prefer the same `actions/github-script` + PAT merge approach already used
    in `qa-question-auto-pr.yml` rather than `gh pr merge` with
    `GITHUB_TOKEN`.
- [ ] Merge method: squash.

### 2c. Bot loop prevention

- [ ] **Do not automatically add sync actor to `contract-trigger` ignore regex.**
      `palette-crimson/` sync merges should normally trigger downstream
      assessments (especially crimson).
- [ ] Verify merge actor behavior:
  - If sync PR merges as `github-actions[bot]`, current ignore regex may skip
    trigger jobs.
  - Prefer merge identity that is not ignored, or adjust actor policy
    deliberately after validating no-loop behavior.

---

## Phase 3: Restrict `palette-answer-consumer`

The `palette-answer-consumer` workflow currently allows edits to
`palette-crimson/`. Under the palette-as-author model, only palette should
modify those contracts. The answer consumer should be restricted to editing
contracts that palette *consumes* (`citrus-palette/`, `orange-palette/`).

### 3a. Workflow file changes

**File:** `contracts/.github/workflows/palette-answer-consumer.yml`

- [ ] **Line 55** — Remove `palette-crimson` from git status check:
  ```bash
  # Before:
  if [[ -n "$(git status --porcelain -- citrus-palette orange-palette palette-crimson docs)" ]]; then
  # After:
  if [[ -n "$(git status --porcelain -- citrus-palette orange-palette docs)" ]]; then
  ```
- [ ] **Line 74** — Remove `palette-crimson/` from PR body scope description:
  ```
  # Before:
  Scope: `citrus-palette/`, `orange-palette/`, `palette-crimson/`, `docs/`
  # After:
  Scope: `citrus-palette/`, `orange-palette/`, `docs/`
  ```
- [ ] **Lines 77-81** — Remove `palette-crimson/` from `add-paths`:
  ```yaml
  # Before:
  add-paths: |
    citrus-palette/
    orange-palette/
    palette-crimson/
    docs/
  # After:
  add-paths: |
    citrus-palette/
    orange-palette/
    docs/
  ```

### 3b. Agent prompt changes

**File:** `contracts/automation/run_palette_answer_consumer.sh`

- [ ] **Lines 130-133** — Remove `palette-crimson/` from the task scope
      instruction in the prompt:
  ```
  # Before:
  2) Scope edits to contract documentation only:
     - citrus-palette/
     - orange-palette/
     - palette-crimson/ (only if directly impacted by answer content)
  # After:
  2) Scope edits to contract documentation only:
     - citrus-palette/
     - orange-palette/
  Note: palette-crimson/ is authored by the palette repo and synced via CI.
  Do not edit palette-crimson/ files. If an answer suggests changes to
  palette-crimson contracts, note the recommendation in the PR body instead.
  ```

### 3c. No changes needed

- **`palette-clarify-scout.yml`** — already restricted to
  `qa/questions/palette-to-citrus/` and `qa/questions/palette-to-orange/`.
  Does not touch `palette-crimson/`.
- **`contract-trigger.yml`** — read-only assessment. No edits to any directory.
- **`question-router.yml`** — only edits `qa/answers/`. No contract files.

---

## Phase 4: Verification

- [ ] After palette-side setup (Phase 1), run `check_contract_freshness.py`
      to verify all contracts are detected at new paths.
- [ ] After first CI sync (Phase 2), verify contracts repo `palette-crimson/`
      matches palette `docs/palette-crimson/` exactly (including
      `contract-meta` blocks).
- [ ] After answer-consumer restriction (Phase 3), trigger a test run of
      `palette-answer-consumer` and verify it does NOT modify
      `palette-crimson/` files.
- [ ] Verify `contract-trigger` still fires for the crimson agent when
      `palette-crimson/` files change via sync PR.
- [ ] Verify the clarify loop still works end-to-end (contract change →
      question → answer → consumer proposes update to `citrus-palette/` only).

---

## Blocking relationships

- Phase 1 (palette directory setup) can proceed independently.
- Phase 2 (CI sync) requires Phase 1 to be complete.
- Phase 3 (answer-consumer restriction) can proceed independently of Phases 1-2
  but should land before the first sync PR to avoid the answer-consumer
  overwriting synced content.
- Phase 4 (verification) requires all prior phases.

## Future work

- If citrus or orange adopt the same pattern (authoring their own contracts in
  their repos), add equivalent sync workflows and restrict their
  answer-consumers similarly.
- Consider whether `docs/` edits by the answer-consumer should also be
  restricted or scoped more narrowly.

## Related docs

- `docs/keypoints_pipeline_inline_registry_report.md` — references contract
  docs for the step status API
- `docs/recording_step_status_parallel_agents_contract.md` — step status
  contract (palette-internal, not affected by this migration)
- `contracts/docs/contract-trigger-matrix.md` — watch matrix configuration
- `contracts/docs/palette-agent-automation.md` — palette agent automation guide
- `contracts/docs/palette-clarify-loop-operations.md` — clarify loop docs
